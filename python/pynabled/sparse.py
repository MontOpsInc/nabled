"""Sparse carriers and sparse API wrappers for pynabled."""

from __future__ import annotations

from typing import Any

import numpy as np

from pynabled._pynabled import (
    sparse_matmat_dense as _sparse_matmat_dense_raw,
)
from pynabled._pynabled import (
    sparse_matvec as _sparse_matvec_raw,
)
from pynabled._pynabled import (
    sparse_pcg_solve as _sparse_pcg_solve_raw,
)
from pynabled._pynabled import (
    sparse_transpose as _sparse_transpose_raw,
)

from ._pynabled import (
    sparse_jacobi_solve as _sparse_jacobi_solve_raw,
)


def _normalize_shape(shape: Any) -> tuple[int, int]:
    try:
        nrows, ncols = shape
    except Exception as exc:  # pragma: no cover - defensive shape unpacking
        raise TypeError("shape must be a length-2 tuple of integers") from exc
    nrows = int(nrows)
    ncols = int(ncols)
    if nrows <= 0 or ncols <= 0:
        raise ValueError("shape dimensions must be positive")
    return nrows, ncols


def _normalize_1d(name: str, value: Any, dtype: np.dtype[Any], *, copy: bool) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True) if copy else np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return array


def _normalize_vector(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("vector must be a 1D array")
    return array


def _normalize_dense(dense: Any) -> np.ndarray:
    array = np.asarray(dense, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("dense operand must be a 2D array")
    return array


class CsrMatrix:
    """Canonical Python carrier for CSR sparse matrices in `pynabled`.

    This carrier normalizes indices to `int64` and values to `float64` for the current sparse
    Python surface. SciPy-compatible CSR objects are accepted through `from_scipy()` or the public
    sparse wrappers.
    """

    __slots__ = ("shape", "indptr", "indices", "data")
    __array_priority__ = 1000

    def __init__(
        self,
        shape: tuple[int, int],
        indptr: Any,
        indices: Any,
        data: Any,
        *,
        copy: bool = False,
    ) -> None:
        nrows, ncols = _normalize_shape(shape)
        indptr_array = _normalize_1d("indptr", indptr, np.int64, copy=copy)
        indices_array = _normalize_1d("indices", indices, np.int64, copy=copy)
        data_array = _normalize_1d("data", data, np.float64, copy=copy)
        if indptr_array.shape[0] != nrows + 1:
            raise ValueError("indptr length must equal nrows + 1")
        if indices_array.shape[0] != data_array.shape[0]:
            raise ValueError("indices and data must have matching lengths")
        if indptr_array[0] != 0:
            raise ValueError("indptr must start at 0")
        if indptr_array[-1] != indices_array.shape[0]:
            raise ValueError("indptr terminal offset must equal nnz")
        if np.any(indptr_array[1:] < indptr_array[:-1]):
            raise ValueError("indptr must be non-decreasing")
        if np.any(indices_array < 0) or np.any(indices_array >= ncols):
            raise ValueError("indices must lie within matrix column bounds")
        self.shape = (nrows, ncols)
        self.indptr = indptr_array
        self.indices = indices_array
        self.data = data_array

    @classmethod
    def from_components(
        cls,
        nrows: int,
        ncols: int,
        indptr: Any,
        indices: Any,
        data: Any,
        *,
        copy: bool = False,
    ) -> "CsrMatrix":
        return cls((nrows, ncols), indptr, indices, data, copy=copy)

    @classmethod
    def from_scipy(cls, matrix: Any, *, copy: bool = False) -> "CsrMatrix":
        if isinstance(matrix, cls):
            return matrix.copy() if copy else matrix
        if getattr(matrix, "format", None) != "csr":
            tocsr = getattr(matrix, "tocsr", None)
            if tocsr is None:
                raise TypeError(
                    "expected pynabled.CsrMatrix or a scipy.sparse-compatible object with tocsr()"
                )
            matrix = tocsr(copy=copy)
        for attribute in ("shape", "indptr", "indices", "data"):
            if not hasattr(matrix, attribute):
                raise TypeError(
                    "expected pynabled.CsrMatrix or a scipy.sparse-compatible CSR object"
                )
        return cls(matrix.shape, matrix.indptr, matrix.indices, matrix.data, copy=copy)

    @property
    def nrows(self) -> int:
        return self.shape[0]

    @property
    def ncols(self) -> int:
        return self.shape[1]

    @property
    def nnz(self) -> int:
        return int(self.data.shape[0])

    @property
    def T(self) -> "CsrMatrix":
        return self.transpose()

    def copy(self) -> "CsrMatrix":
        return CsrMatrix(self.shape, self.indptr, self.indices, self.data, copy=True)

    def to_components(self) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        return self.nrows, self.ncols, self.indptr, self.indices, self.data

    def to_scipy(self) -> Any:
        try:
            from scipy import sparse as scipy_sparse
        except ImportError as exc:  # pragma: no cover - depends on optional scipy install
            raise ImportError("scipy is required for CsrMatrix.to_scipy()") from exc
        return scipy_sparse.csr_matrix(
            (self.data, self.indices, self.indptr),
            shape=self.shape,
            copy=False,
        )

    def matvec(self, vector: Any) -> np.ndarray:
        return sparse_matvec(self, vector)

    def matmat_dense(self, dense: Any) -> np.ndarray:
        return sparse_matmat_dense(self, dense)

    def transpose(self) -> "CsrMatrix":
        return sparse_transpose(self)

    def jacobi_solve(
        self,
        rhs: Any,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return sparse_jacobi_solve(self, rhs, tolerance=tolerance, max_iterations=max_iterations)

    def pcg_solve(
        self,
        rhs: Any,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return sparse_pcg_solve(self, rhs, tolerance=tolerance, max_iterations=max_iterations)

    def __matmul__(self, other: Any) -> np.ndarray:
        other_array = np.asarray(other)
        if other_array.ndim == 1:
            return self.matvec(other_array)
        if other_array.ndim == 2:
            return self.matmat_dense(other_array)
        return NotImplemented

    def __repr__(self) -> str:
        return f"CsrMatrix(shape={self.shape}, nnz={self.nnz}, dtype={self.data.dtype})"


def _coerce_csr_matrix(matrix: Any) -> CsrMatrix:
    if isinstance(matrix, CsrMatrix):
        return matrix
    return CsrMatrix.from_scipy(matrix)


def sparse_matvec(matrix: Any, vector: Any) -> np.ndarray:
    csr = _coerce_csr_matrix(matrix)
    return _sparse_matvec_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_vector(vector),
    )


def sparse_matmat_dense(matrix: Any, dense: Any) -> np.ndarray:
    csr = _coerce_csr_matrix(matrix)
    return _sparse_matmat_dense_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_dense(dense),
    )


def sparse_transpose(matrix: Any) -> CsrMatrix:
    csr = _coerce_csr_matrix(matrix)
    return CsrMatrix.from_components(
        *_sparse_transpose_raw(csr.nrows, csr.ncols, csr.indptr, csr.indices, csr.data)
    )


def sparse_jacobi_solve(
    matrix: Any,
    rhs: Any,
    *,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> np.ndarray:
    csr = _coerce_csr_matrix(matrix)
    return _sparse_jacobi_solve_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_vector(rhs),
        tolerance,
        max_iterations,
    )


def sparse_pcg_solve(
    matrix: Any,
    rhs: Any,
    *,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> np.ndarray:
    csr = _coerce_csr_matrix(matrix)
    return _sparse_pcg_solve_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_vector(rhs),
        tolerance,
        max_iterations,
    )


__all__ = [
    "CsrMatrix",
    "sparse_matvec",
    "sparse_matmat_dense",
    "sparse_transpose",
    "sparse_jacobi_solve",
    "sparse_pcg_solve",
]
