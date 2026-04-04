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

_REAL_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_INDEX_DTYPES = (np.dtype(np.int32), np.dtype(np.int64))
_INT32_MAX = np.iinfo(np.int32).max


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


def _normalize_explicit_dtype(
    name: str, dtype: Any, allowed: tuple[np.dtype[Any], ...]
) -> np.dtype[Any]:
    resolved = np.dtype(dtype)
    if resolved not in allowed:
        allowed_names = " or ".join(candidate.name for candidate in allowed)
        raise TypeError(f"{name} must be {allowed_names}")
    return resolved


def _require_c_contiguous(name: str, array: np.ndarray) -> np.ndarray:
    if not array.flags.c_contiguous:
        raise ValueError(
            f"{name} must be C-contiguous; pass copy=True or an explicit normalization dtype"
        )
    return array


def _normalize_1d(name: str, value: Any, dtype: np.dtype[Any], *, copy: bool) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True) if copy else np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return _require_c_contiguous(name, array)


def _normalize_real_array(
    name: str,
    value: Any,
    *,
    ndim: int,
    copy: bool,
    dtype: np.dtype[Any] | None = None,
    require_c_contiguous: bool = False,
    allow_cast: bool = False,
    mismatch_message: str | None = None,
) -> np.ndarray:
    if dtype is None:
        array = np.array(value, copy=True) if copy else np.asarray(value)
    elif allow_cast:
        array = np.array(value, dtype=dtype, copy=True) if copy else np.asarray(value, dtype=dtype)
    else:
        array = np.array(value, copy=True) if copy else np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array")
    if dtype is None:
        if array.dtype not in _REAL_DTYPES:
            raise TypeError(f"{name} must have dtype float32 or float64")
    elif array.dtype != dtype:
        raise TypeError(mismatch_message or f"{name} must have dtype {dtype.name}")
    if require_c_contiguous:
        return _require_c_contiguous(name, array)
    return array


def _resolve_data_dtype(dtype: Any | None) -> np.dtype[Any] | None:
    if dtype is None:
        return None
    return _normalize_explicit_dtype("dtype", dtype, _REAL_DTYPES)


def _resolve_index_dtype(
    indptr: Any,
    indices: Any,
    *,
    ncols: int,
    nnz: int,
    index_dtype: Any | None,
) -> np.dtype[Any]:
    if index_dtype is not None:
        return _normalize_explicit_dtype("index_dtype", index_dtype, _INDEX_DTYPES)

    if isinstance(indptr, np.ndarray) and indptr.dtype not in _INDEX_DTYPES:
        raise TypeError(
            "indptr must have dtype int32 or int64; pass index_dtype=... to normalize explicitly"
        )
    if isinstance(indices, np.ndarray) and indices.dtype not in _INDEX_DTYPES:
        raise TypeError(
            "indices must have dtype int32 or int64; pass index_dtype=... to normalize explicitly"
        )

    indptr_dtype = (
        indptr.dtype if isinstance(indptr, np.ndarray) and indptr.dtype in _INDEX_DTYPES else None
    )
    indices_dtype = (
        indices.dtype
        if isinstance(indices, np.ndarray) and indices.dtype in _INDEX_DTYPES
        else None
    )
    if indptr_dtype is not None and indices_dtype is not None:
        if indptr_dtype != indices_dtype:
            raise TypeError(
                "indptr and indices must share dtype int32 or int64; pass index_dtype=... to normalize explicitly"
            )
        return indptr_dtype
    if indptr_dtype is not None:
        return indptr_dtype
    if indices_dtype is not None:
        return indices_dtype
    return np.dtype(np.int32 if max(ncols - 1, nnz) <= _INT32_MAX else np.int64)


def _normalize_vector(vector: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    return _normalize_real_array(
        "vector",
        vector,
        ndim=1,
        copy=False,
        dtype=dtype,
        mismatch_message=f"vector must have dtype {dtype.name} to match sparse matrix data",
    )


def _normalize_rhs(rhs: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    return _normalize_real_array(
        "rhs",
        rhs,
        ndim=1,
        copy=False,
        dtype=dtype,
        mismatch_message=f"rhs must have dtype {dtype.name} to match sparse matrix data",
    )


def _normalize_dense(dense: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    return _normalize_real_array(
        "dense operand",
        dense,
        ndim=2,
        copy=False,
        dtype=dtype,
        mismatch_message=f"dense operand must have dtype {dtype.name} to match sparse matrix data",
    )


class CsrMatrix:
    """Canonical Python carrier for CSR sparse matrices in `pynabled`.

    This carrier preserves `int32` or `int64` index buffers and preserves `float32` / `float64`
    values for the current sparse Python surface. SciPy-compatible CSR objects are accepted through
    `from_scipy()` or the public sparse wrappers.
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
        dtype: Any | None = None,
        index_dtype: Any | None = None,
    ) -> None:
        nrows, ncols = _normalize_shape(shape)
        data_dtype = _resolve_data_dtype(dtype)
        data_array = _normalize_real_array(
            "data",
            data,
            ndim=1,
            copy=copy,
            dtype=data_dtype,
            require_c_contiguous=True,
            allow_cast=data_dtype is not None,
        )
        resolved_index_dtype = _resolve_index_dtype(
            indptr,
            indices,
            ncols=ncols,
            nnz=data_array.shape[0],
            index_dtype=index_dtype,
        )
        indptr_array = _normalize_1d("indptr", indptr, resolved_index_dtype, copy=copy)
        indices_array = _normalize_1d("indices", indices, resolved_index_dtype, copy=copy)
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
        *components: Any,
        copy: bool = False,
        dtype: Any | None = None,
        index_dtype: Any | None = None,
    ) -> "CsrMatrix":
        if len(components) == 4:
            shape, indptr, indices, data = components
        elif len(components) == 5:
            nrows, ncols, indptr, indices, data = components
            shape = (nrows, ncols)
        else:
            raise TypeError(
                "from_components expects (shape, indptr, indices, data) or "
                "(nrows, ncols, indptr, indices, data)"
            )
        return cls(shape, indptr, indices, data, copy=copy, dtype=dtype, index_dtype=index_dtype)

    @classmethod
    def from_scipy(
        cls,
        matrix: Any,
        *,
        copy: bool = False,
        dtype: Any | None = None,
        index_dtype: Any | None = None,
    ) -> "CsrMatrix":
        if isinstance(matrix, cls):
            if dtype is None and index_dtype is None and not copy:
                return matrix
            return cls(
                matrix.shape,
                matrix.indptr,
                matrix.indices,
                matrix.data,
                copy=copy,
                dtype=matrix.dtype if dtype is None else dtype,
                index_dtype=matrix.index_dtype if index_dtype is None else index_dtype,
            )
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
        return cls(
            matrix.shape,
            matrix.indptr,
            matrix.indices,
            matrix.data,
            copy=copy,
            dtype=dtype,
            index_dtype=index_dtype,
        )

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
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def index_dtype(self) -> np.dtype[Any]:
        return self.indptr.dtype

    @property
    def T(self) -> "CsrMatrix":
        return self.transpose()

    def copy(self) -> "CsrMatrix":
        return CsrMatrix(
            self.shape,
            self.indptr,
            self.indices,
            self.data,
            copy=True,
            dtype=self.dtype,
            index_dtype=self.index_dtype,
        )

    def astype(self, dtype: Any, *, copy: bool = False) -> "CsrMatrix":
        resolved_dtype = _resolve_data_dtype(dtype)
        if resolved_dtype == self.dtype and not copy:
            return self
        return CsrMatrix(
            self.shape,
            self.indptr,
            self.indices,
            self.data,
            copy=copy,
            dtype=resolved_dtype,
            index_dtype=self.index_dtype,
        )

    def with_index_dtype(self, index_dtype: Any, *, copy: bool = False) -> "CsrMatrix":
        resolved_index_dtype = _normalize_explicit_dtype("index_dtype", index_dtype, _INDEX_DTYPES)
        if resolved_index_dtype == self.index_dtype and not copy:
            return self
        return CsrMatrix(
            self.shape,
            self.indptr,
            self.indices,
            self.data,
            copy=copy,
            dtype=self.dtype,
            index_dtype=resolved_index_dtype,
        )

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
        return (
            f"CsrMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"dtype={self.data.dtype}, index_dtype={self.index_dtype})"
        )


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
        _normalize_vector(vector, dtype=csr.data.dtype),
    )


def sparse_matmat_dense(matrix: Any, dense: Any) -> np.ndarray:
    csr = _coerce_csr_matrix(matrix)
    return _sparse_matmat_dense_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_dense(dense, dtype=csr.data.dtype),
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
        _normalize_rhs(rhs, dtype=csr.data.dtype),
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
        _normalize_rhs(rhs, dtype=csr.data.dtype),
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
