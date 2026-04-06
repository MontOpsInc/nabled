"""Sparse carriers and sparse API wrappers for pynabled."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pynabled._pynabled import (
    _SparseIC0Factorization as _RawSparseIC0Factorization,
)
from pynabled._pynabled import (
    _SparseILDL0Factorization as _RawSparseILDL0Factorization,
)
from pynabled._pynabled import (
    _SparseILU0Factorization as _RawSparseILU0Factorization,
)
from pynabled._pynabled import (
    _SparseILUKFactorization as _RawSparseILUKFactorization,
)
from pynabled._pynabled import (
    _SparseILUTFactorization as _RawSparseILUTFactorization,
)
from pynabled._pynabled import (
    _SparseJacobiPreconditioner as _RawSparseJacobiPreconditioner,
)
from pynabled._pynabled import (
    _SparseLUFactorization as _RawSparseLUFactorization,
)
from pynabled._pynabled import (
    sparse_bicgstab_solve as _sparse_bicgstab_solve_raw,
)
from pynabled._pynabled import (
    sparse_conjugate_gradient_solve as _sparse_conjugate_gradient_solve_raw,
)
from pynabled._pynabled import (
    sparse_coo_to_csr as _sparse_coo_to_csr_raw,
)
from pynabled._pynabled import (
    sparse_csc_to_csr as _sparse_csc_to_csr_raw,
)
from pynabled._pynabled import (
    sparse_csr_to_csc as _sparse_csr_to_csc_raw,
)
from pynabled._pynabled import (
    sparse_gauss_seidel_solve as _sparse_gauss_seidel_solve_raw,
)
from pynabled._pynabled import (
    sparse_ic0_factor as _sparse_ic0_factor_raw,
)
from pynabled._pynabled import (
    sparse_ildl0_factor as _sparse_ildl0_factor_raw,
)
from pynabled._pynabled import (
    sparse_ilu0_factor as _sparse_ilu0_factor_raw,
)
from pynabled._pynabled import (
    sparse_iluk_factor as _sparse_iluk_factor_raw,
)
from pynabled._pynabled import (
    sparse_ilut_factor as _sparse_ilut_factor_raw,
)
from pynabled._pynabled import (
    sparse_jacobi_preconditioner as _sparse_jacobi_preconditioner_raw,
)
from pynabled._pynabled import (
    sparse_lu_factor as _sparse_lu_factor_raw,
)
from pynabled._pynabled import (
    sparse_matmat_dense as _sparse_matmat_dense_raw,
)
from pynabled._pynabled import (
    sparse_matmat_sparse as _sparse_matmat_sparse_raw,
)
from pynabled._pynabled import (
    sparse_matvec as _sparse_matvec_raw,
)
from pynabled._pynabled import (
    sparse_matvec_csc as _sparse_matvec_csc_raw,
)
from pynabled._pynabled import (
    sparse_pcg_solve as _sparse_pcg_solve_raw,
)
from pynabled._pynabled import (
    sparse_pcg_ic0_solve as _sparse_pcg_ic0_solve_raw,
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


def _resolve_coordinate_index_dtype(
    row_indices: Any,
    col_indices: Any,
    *,
    nrows: int,
    ncols: int,
    nnz: int,
    index_dtype: Any | None,
) -> np.dtype[Any]:
    if index_dtype is not None:
        return _normalize_explicit_dtype("index_dtype", index_dtype, _INDEX_DTYPES)

    if isinstance(row_indices, np.ndarray) and row_indices.dtype not in _INDEX_DTYPES:
        raise TypeError(
            "row_indices must have dtype int32 or int64; pass index_dtype=... to normalize explicitly"
        )
    if isinstance(col_indices, np.ndarray) and col_indices.dtype not in _INDEX_DTYPES:
        raise TypeError(
            "col_indices must have dtype int32 or int64; pass index_dtype=... to normalize explicitly"
        )

    row_dtype = (
        row_indices.dtype
        if isinstance(row_indices, np.ndarray) and row_indices.dtype in _INDEX_DTYPES
        else None
    )
    col_dtype = (
        col_indices.dtype
        if isinstance(col_indices, np.ndarray) and col_indices.dtype in _INDEX_DTYPES
        else None
    )
    if row_dtype is not None and col_dtype is not None:
        if row_dtype != col_dtype:
            raise TypeError(
                "row_indices and col_indices must share dtype int32 or int64; pass index_dtype=... to normalize explicitly"
            )
        return row_dtype
    if row_dtype is not None:
        return row_dtype
    if col_dtype is not None:
        return col_dtype
    max_index = max(nrows - 1, ncols - 1, nnz)
    return np.dtype(np.int32 if max_index <= _INT32_MAX else np.int64)


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


def _normalize_rhs_matrix(rhs: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    return _normalize_real_array(
        "rhs",
        rhs,
        ndim=2,
        copy=False,
        dtype=dtype,
        mismatch_message=f"rhs must have dtype {dtype.name} to match sparse factorization data",
    )


@dataclass(slots=True)
class ILUTConfig:
    drop_tolerance: float
    max_fill: int

    @classmethod
    def conservative(cls) -> "ILUTConfig":
        return cls(drop_tolerance=1e-6, max_fill=8)

    @classmethod
    def balanced(cls) -> "ILUTConfig":
        return cls(drop_tolerance=1e-8, max_fill=16)

    @classmethod
    def aggressive(cls) -> "ILUTConfig":
        return cls(drop_tolerance=1e-10, max_fill=32)

    @classmethod
    def for_dimension(cls, dimension: int) -> "ILUTConfig":
        fill = 8 if dimension <= 32 else 16 if dimension <= 256 else 32
        return cls(drop_tolerance=1e-8, max_fill=min(fill, max(int(dimension), 1)))


@dataclass(slots=True)
class ILUKConfig:
    level_of_fill: int

    @classmethod
    def conservative(cls) -> "ILUKConfig":
        return cls(level_of_fill=0)

    @classmethod
    def balanced(cls) -> "ILUKConfig":
        return cls(level_of_fill=1)

    @classmethod
    def aggressive(cls) -> "ILUKConfig":
        return cls(level_of_fill=2)


def _coerce_ilut_config(
    matrix: "CsrMatrix",
    *,
    config: ILUTConfig | None,
    drop_tolerance: float | None,
    max_fill: int | None,
) -> ILUTConfig:
    if config is not None and (drop_tolerance is not None or max_fill is not None):
        raise TypeError("pass either config=... or explicit drop_tolerance/max_fill, not both")
    if config is not None:
        return config
    base = ILUTConfig.balanced()
    if drop_tolerance is None and max_fill is None:
        return base
    return ILUTConfig(
        drop_tolerance=base.drop_tolerance if drop_tolerance is None else float(drop_tolerance),
        max_fill=base.max_fill if max_fill is None else int(max_fill),
    )


def _coerce_iluk_config(
    *,
    config: ILUKConfig | None,
    level_of_fill: int | None,
) -> ILUKConfig:
    if config is not None and level_of_fill is not None:
        raise TypeError("pass either config=... or level_of_fill=..., not both")
    if config is not None:
        return config
    return (
        ILUKConfig.balanced()
        if level_of_fill is None
        else ILUKConfig(level_of_fill=int(level_of_fill))
    )


class JacobiPreconditioner:
    __slots__ = ("_raw", "_inverse_diagonal")

    def __init__(self, raw: _RawSparseJacobiPreconditioner) -> None:
        self._raw = raw
        self._inverse_diagonal: np.ndarray | None = None

    @property
    def inverse_diagonal(self) -> np.ndarray:
        if self._inverse_diagonal is None:
            self._inverse_diagonal = self._raw.inverse_diagonal
        return self._inverse_diagonal

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.inverse_diagonal.dtype

    def apply(self, rhs: Any) -> np.ndarray:
        return self._raw.apply(_normalize_rhs(rhs, dtype=self.dtype))

    def __repr__(self) -> str:
        return f"JacobiPreconditioner(size={self.inverse_diagonal.shape[0]}, dtype={self.dtype})"


class ILU0Factorization:
    __slots__ = ("matrix", "_raw", "_l", "_u")

    def __init__(self, matrix: "CsrMatrix", raw: _RawSparseILU0Factorization) -> None:
        self.matrix = matrix
        self._raw = raw
        self._l: CsrMatrix | None = None
        self._u: CsrMatrix | None = None

    @property
    def l(self) -> "CsrMatrix":
        if self._l is None:
            self._l = CsrMatrix.from_components(*self._raw.l_parts())
        return self._l

    @property
    def u(self) -> "CsrMatrix":
        if self._u is None:
            self._u = CsrMatrix.from_components(*self._raw.u_parts())
        return self._u

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.matrix.dtype

    def apply(self, rhs: Any) -> np.ndarray:
        return self._raw.apply(_normalize_rhs(rhs, dtype=self.dtype))

    def gmres_solve(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.gmres_solve(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def gmres_solve_multiple(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.gmres_solve_multiple(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs_matrix(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def bicgstab_solve(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.bicgstab_solve(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def bicgstab_solve_multiple(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.bicgstab_solve_multiple(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs_matrix(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def __repr__(self) -> str:
        return f"ILU0Factorization(shape={self.matrix.shape}, dtype={self.dtype})"


class ILUTFactorization:
    __slots__ = ("matrix", "_raw", "_l", "_u")

    def __init__(self, matrix: "CsrMatrix", raw: _RawSparseILUTFactorization) -> None:
        self.matrix = matrix
        self._raw = raw
        self._l: CsrMatrix | None = None
        self._u: CsrMatrix | None = None

    @property
    def l(self) -> "CsrMatrix":
        if self._l is None:
            self._l = CsrMatrix.from_components(*self._raw.l_parts())
        return self._l

    @property
    def u(self) -> "CsrMatrix":
        if self._u is None:
            self._u = CsrMatrix.from_components(*self._raw.u_parts())
        return self._u

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.matrix.dtype

    def apply(self, rhs: Any) -> np.ndarray:
        return self._raw.apply(_normalize_rhs(rhs, dtype=self.dtype))

    def gmres_solve(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.gmres_solve(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def gmres_solve_multiple(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.gmres_solve_multiple(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs_matrix(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def bicgstab_solve(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.bicgstab_solve(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def bicgstab_solve_multiple(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.bicgstab_solve_multiple(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs_matrix(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def __repr__(self) -> str:
        return f"ILUTFactorization(shape={self.matrix.shape}, dtype={self.dtype})"


class ILUKFactorization:
    __slots__ = ("matrix", "_raw", "_l", "_u")

    def __init__(self, matrix: "CsrMatrix", raw: _RawSparseILUKFactorization) -> None:
        self.matrix = matrix
        self._raw = raw
        self._l: CsrMatrix | None = None
        self._u: CsrMatrix | None = None

    @property
    def l(self) -> "CsrMatrix":
        if self._l is None:
            self._l = CsrMatrix.from_components(*self._raw.l_parts())
        return self._l

    @property
    def u(self) -> "CsrMatrix":
        if self._u is None:
            self._u = CsrMatrix.from_components(*self._raw.u_parts())
        return self._u

    @property
    def level_of_fill(self) -> int:
        return int(self._raw.level_of_fill)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.matrix.dtype

    def apply(self, rhs: Any) -> np.ndarray:
        return self._raw.apply(_normalize_rhs(rhs, dtype=self.dtype))

    def gmres_solve(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.gmres_solve(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def gmres_solve_multiple(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.gmres_solve_multiple(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs_matrix(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def bicgstab_solve(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.bicgstab_solve(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def bicgstab_solve_multiple(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.bicgstab_solve_multiple(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs_matrix(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def __repr__(self) -> str:
        return (
            f"ILUKFactorization(shape={self.matrix.shape}, level_of_fill={self.level_of_fill}, "
            f"dtype={self.dtype})"
        )


class IC0Factorization:
    __slots__ = ("matrix", "_raw", "_l", "_l_transpose")

    def __init__(self, matrix: "CsrMatrix", raw: _RawSparseIC0Factorization) -> None:
        self.matrix = matrix
        self._raw = raw
        self._l: CsrMatrix | None = None
        self._l_transpose: CsrMatrix | None = None

    @property
    def l(self) -> "CsrMatrix":
        if self._l is None:
            self._l = CsrMatrix.from_components(*self._raw.l_parts())
        return self._l

    @property
    def l_transpose(self) -> "CsrMatrix":
        if self._l_transpose is None:
            self._l_transpose = CsrMatrix.from_components(*self._raw.l_transpose_parts())
        return self._l_transpose

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.matrix.dtype

    def apply(self, rhs: Any) -> np.ndarray:
        return self._raw.apply(_normalize_rhs(rhs, dtype=self.dtype))

    def pcg_solve(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.pcg_solve(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def __repr__(self) -> str:
        return f"IC0Factorization(shape={self.matrix.shape}, dtype={self.dtype})"


class ILDL0Factorization:
    __slots__ = ("matrix", "_raw", "_l", "_l_transpose", "_d")

    def __init__(self, matrix: "CsrMatrix", raw: _RawSparseILDL0Factorization) -> None:
        self.matrix = matrix
        self._raw = raw
        self._l: CsrMatrix | None = None
        self._l_transpose: CsrMatrix | None = None
        self._d: np.ndarray | None = None

    @property
    def l(self) -> "CsrMatrix":
        if self._l is None:
            self._l = CsrMatrix.from_components(*self._raw.l_parts())
        return self._l

    @property
    def l_transpose(self) -> "CsrMatrix":
        if self._l_transpose is None:
            self._l_transpose = CsrMatrix.from_components(*self._raw.l_transpose_parts())
        return self._l_transpose

    @property
    def d(self) -> np.ndarray:
        if self._d is None:
            self._d = self._raw.d
        return self._d

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.matrix.dtype

    def apply(self, rhs: Any) -> np.ndarray:
        return self._raw.apply(_normalize_rhs(rhs, dtype=self.dtype))

    def gmres_solve(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.gmres_solve(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def gmres_solve_multiple(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.gmres_solve_multiple(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs_matrix(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def bicgstab_solve(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.bicgstab_solve(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def bicgstab_solve_multiple(
        self,
        rhs: Any,
        *,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return self._raw.bicgstab_solve_multiple(
            self.matrix.nrows,
            self.matrix.ncols,
            self.matrix.indptr,
            self.matrix.indices,
            self.matrix.data,
            _normalize_rhs_matrix(rhs, dtype=self.dtype),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def __repr__(self) -> str:
        return f"ILDL0Factorization(shape={self.matrix.shape}, dtype={self.dtype})"


class SparseLUFactorization:
    __slots__ = ("matrix", "_raw", "_l", "_u", "_permutation")

    def __init__(self, matrix: "CsrMatrix", raw: _RawSparseLUFactorization) -> None:
        self.matrix = matrix
        self._raw = raw
        self._l: CsrMatrix | None = None
        self._u: CsrMatrix | None = None
        self._permutation: np.ndarray | None = None

    @property
    def l(self) -> "CsrMatrix":
        if self._l is None:
            self._l = CsrMatrix.from_components(*self._raw.l_parts())
        return self._l

    @property
    def u(self) -> "CsrMatrix":
        if self._u is None:
            self._u = CsrMatrix.from_components(*self._raw.u_parts())
        return self._u

    @property
    def permutation(self) -> np.ndarray:
        if self._permutation is None:
            self._permutation = self._raw.permutation
        return self._permutation

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.matrix.dtype

    def solve(self, rhs: Any) -> np.ndarray:
        return self._raw.solve(_normalize_rhs(rhs, dtype=self.dtype))

    def solve_multiple(self, rhs: Any) -> np.ndarray:
        return self._raw.solve_multiple(_normalize_rhs_matrix(rhs, dtype=self.dtype))

    def __repr__(self) -> str:
        return f"SparseLUFactorization(shape={self.matrix.shape}, dtype={self.dtype})"


class CscMatrix:
    """Canonical Python carrier for CSC sparse matrices in `pynabled`."""

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
        if indptr_array.shape[0] != ncols + 1:
            raise ValueError("indptr length must equal ncols + 1")
        if indices_array.shape[0] != data_array.shape[0]:
            raise ValueError("indices and data must have matching lengths")
        if indptr_array[0] != 0:
            raise ValueError("indptr must start at 0")
        if indptr_array[-1] != indices_array.shape[0]:
            raise ValueError("indptr terminal offset must equal nnz")
        if np.any(indptr_array[1:] < indptr_array[:-1]):
            raise ValueError("indptr must be non-decreasing")
        if np.any(indices_array < 0) or np.any(indices_array >= nrows):
            raise ValueError("indices must lie within matrix row bounds")
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
    ) -> "CscMatrix":
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
    ) -> "CscMatrix":
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
        if getattr(matrix, "format", None) != "csc":
            tocsc = getattr(matrix, "tocsc", None)
            if tocsc is None:
                raise TypeError(
                    "expected pynabled.CscMatrix or a scipy.sparse-compatible object with tocsc()"
                )
            matrix = tocsc(copy=copy)
        for attribute in ("shape", "indptr", "indices", "data"):
            if not hasattr(matrix, attribute):
                raise TypeError(
                    "expected pynabled.CscMatrix or a scipy.sparse-compatible CSC object"
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

    def copy(self) -> "CscMatrix":
        return CscMatrix(
            self.shape,
            self.indptr,
            self.indices,
            self.data,
            copy=True,
            dtype=self.dtype,
            index_dtype=self.index_dtype,
        )

    def astype(self, dtype: Any, *, copy: bool = False) -> "CscMatrix":
        resolved_dtype = _resolve_data_dtype(dtype)
        if resolved_dtype == self.dtype and not copy:
            return self
        return CscMatrix(
            self.shape,
            self.indptr,
            self.indices,
            self.data,
            copy=copy,
            dtype=resolved_dtype,
            index_dtype=self.index_dtype,
        )

    def with_index_dtype(self, index_dtype: Any, *, copy: bool = False) -> "CscMatrix":
        resolved_index_dtype = _normalize_explicit_dtype("index_dtype", index_dtype, _INDEX_DTYPES)
        if resolved_index_dtype == self.index_dtype and not copy:
            return self
        return CscMatrix(
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
            raise ImportError("scipy is required for CscMatrix.to_scipy()") from exc
        return scipy_sparse.csc_matrix(
            (self.data, self.indices, self.indptr),
            shape=self.shape,
            copy=False,
        )

    def to_csr(self) -> "CsrMatrix":
        return CsrMatrix.from_components(
            *_sparse_csc_to_csr_raw(self.nrows, self.ncols, self.indptr, self.indices, self.data)
        )

    def matvec(self, vector: Any) -> np.ndarray:
        return sparse_matvec_csc(self, vector)

    def matmat_dense(self, dense: Any) -> np.ndarray:
        return self.to_csr().matmat_dense(dense)

    def transpose(self) -> "CsrMatrix":
        return self.to_csr().transpose()

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, (CsrMatrix, CscMatrix, CooMatrix)):
            return sparse_matmat_sparse(self, other)
        other_array = np.asarray(other)
        if other_array.ndim == 1:
            return self.matvec(other_array)
        if other_array.ndim == 2:
            return self.matmat_dense(other_array)
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"CscMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"dtype={self.data.dtype}, index_dtype={self.index_dtype})"
        )


class CooMatrix:
    """Canonical Python carrier for COO sparse matrices in `pynabled`."""

    __slots__ = ("shape", "row_indices", "col_indices", "data")
    __array_priority__ = 1000

    def __init__(
        self,
        shape: tuple[int, int],
        row_indices: Any,
        col_indices: Any,
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
        resolved_index_dtype = _resolve_coordinate_index_dtype(
            row_indices,
            col_indices,
            nrows=nrows,
            ncols=ncols,
            nnz=data_array.shape[0],
            index_dtype=index_dtype,
        )
        row_indices_array = _normalize_1d(
            "row_indices", row_indices, resolved_index_dtype, copy=copy
        )
        col_indices_array = _normalize_1d(
            "col_indices", col_indices, resolved_index_dtype, copy=copy
        )
        if row_indices_array.shape[0] != data_array.shape[0]:
            raise ValueError("row_indices and data must have matching lengths")
        if col_indices_array.shape[0] != data_array.shape[0]:
            raise ValueError("col_indices and data must have matching lengths")
        if np.any(row_indices_array < 0) or np.any(row_indices_array >= nrows):
            raise ValueError("row_indices must lie within matrix row bounds")
        if np.any(col_indices_array < 0) or np.any(col_indices_array >= ncols):
            raise ValueError("col_indices must lie within matrix column bounds")
        self.shape = (nrows, ncols)
        self.row_indices = row_indices_array
        self.col_indices = col_indices_array
        self.data = data_array

    @classmethod
    def from_components(
        cls,
        *components: Any,
        copy: bool = False,
        dtype: Any | None = None,
        index_dtype: Any | None = None,
    ) -> "CooMatrix":
        if len(components) == 4:
            shape, row_indices, col_indices, data = components
        elif len(components) == 5:
            nrows, ncols, row_indices, col_indices, data = components
            shape = (nrows, ncols)
        else:
            raise TypeError(
                "from_components expects (shape, row_indices, col_indices, data) or "
                "(nrows, ncols, row_indices, col_indices, data)"
            )
        return cls(
            shape,
            row_indices,
            col_indices,
            data,
            copy=copy,
            dtype=dtype,
            index_dtype=index_dtype,
        )

    @classmethod
    def from_scipy(
        cls,
        matrix: Any,
        *,
        copy: bool = False,
        dtype: Any | None = None,
        index_dtype: Any | None = None,
    ) -> "CooMatrix":
        if isinstance(matrix, cls):
            if dtype is None and index_dtype is None and not copy:
                return matrix
            return cls(
                matrix.shape,
                matrix.row_indices,
                matrix.col_indices,
                matrix.data,
                copy=copy,
                dtype=matrix.dtype if dtype is None else dtype,
                index_dtype=matrix.index_dtype if index_dtype is None else index_dtype,
            )
        if getattr(matrix, "format", None) != "coo":
            tocoo = getattr(matrix, "tocoo", None)
            if tocoo is None:
                raise TypeError(
                    "expected pynabled.CooMatrix or a scipy.sparse-compatible object with tocoo()"
                )
            matrix = tocoo(copy=copy)
        shape = getattr(matrix, "shape", None)
        row = getattr(matrix, "row", None)
        col = getattr(matrix, "col", None)
        data = getattr(matrix, "data", None)
        if shape is None or row is None or col is None or data is None:
            raise TypeError("expected pynabled.CooMatrix or a scipy.sparse-compatible COO object")
        return cls(shape, row, col, data, copy=copy, dtype=dtype, index_dtype=index_dtype)

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
        return self.row_indices.dtype

    @property
    def T(self) -> "CsrMatrix":
        return self.transpose()

    def copy(self) -> "CooMatrix":
        return CooMatrix(
            self.shape,
            self.row_indices,
            self.col_indices,
            self.data,
            copy=True,
            dtype=self.dtype,
            index_dtype=self.index_dtype,
        )

    def astype(self, dtype: Any, *, copy: bool = False) -> "CooMatrix":
        resolved_dtype = _resolve_data_dtype(dtype)
        if resolved_dtype == self.dtype and not copy:
            return self
        return CooMatrix(
            self.shape,
            self.row_indices,
            self.col_indices,
            self.data,
            copy=copy,
            dtype=resolved_dtype,
            index_dtype=self.index_dtype,
        )

    def with_index_dtype(self, index_dtype: Any, *, copy: bool = False) -> "CooMatrix":
        resolved_index_dtype = _normalize_explicit_dtype("index_dtype", index_dtype, _INDEX_DTYPES)
        if resolved_index_dtype == self.index_dtype and not copy:
            return self
        return CooMatrix(
            self.shape,
            self.row_indices,
            self.col_indices,
            self.data,
            copy=copy,
            dtype=self.dtype,
            index_dtype=resolved_index_dtype,
        )

    def to_components(self) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        return self.nrows, self.ncols, self.row_indices, self.col_indices, self.data

    def to_scipy(self) -> Any:
        try:
            from scipy import sparse as scipy_sparse
        except ImportError as exc:  # pragma: no cover - depends on optional scipy install
            raise ImportError("scipy is required for CooMatrix.to_scipy()") from exc
        return scipy_sparse.coo_matrix(
            (self.data, (self.row_indices, self.col_indices)),
            shape=self.shape,
            copy=False,
        )

    def to_csr(self) -> "CsrMatrix":
        return CsrMatrix.from_components(
            *_sparse_coo_to_csr_raw(
                self.nrows,
                self.ncols,
                self.row_indices,
                self.col_indices,
                self.data,
            )
        )

    def matvec(self, vector: Any) -> np.ndarray:
        return self.to_csr().matvec(vector)

    def matmat_dense(self, dense: Any) -> np.ndarray:
        return self.to_csr().matmat_dense(dense)

    def transpose(self) -> "CsrMatrix":
        return self.to_csr().transpose()

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, (CsrMatrix, CscMatrix, CooMatrix)):
            return sparse_matmat_sparse(self, other)
        other_array = np.asarray(other)
        if other_array.ndim == 1:
            return self.matvec(other_array)
        if other_array.ndim == 2:
            return self.matmat_dense(other_array)
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"CooMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"dtype={self.data.dtype}, index_dtype={self.index_dtype})"
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

    def to_csc(self) -> CscMatrix:
        return CscMatrix.from_components(
            *_sparse_csr_to_csc_raw(self.nrows, self.ncols, self.indptr, self.indices, self.data)
        )

    def matvec(self, vector: Any) -> np.ndarray:
        return sparse_matvec(self, vector)

    def matmat_dense(self, dense: Any) -> np.ndarray:
        return sparse_matmat_dense(self, dense)

    def matmat_sparse(self, other: Any) -> "CsrMatrix":
        return sparse_matmat_sparse(self, other)

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

    def gauss_seidel_solve(
        self,
        rhs: Any,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return sparse_gauss_seidel_solve(
            self, rhs, tolerance=tolerance, max_iterations=max_iterations
        )

    def conjugate_gradient_solve(
        self,
        rhs: Any,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return sparse_conjugate_gradient_solve(
            self, rhs, tolerance=tolerance, max_iterations=max_iterations
        )

    def pcg_ic0_solve(
        self,
        rhs: Any,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return sparse_pcg_ic0_solve(self, rhs, tolerance=tolerance, max_iterations=max_iterations)

    def bicgstab_solve(
        self,
        rhs: Any,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        return sparse_bicgstab_solve(self, rhs, tolerance=tolerance, max_iterations=max_iterations)

    def jacobi_preconditioner(self) -> JacobiPreconditioner:
        return sparse_jacobi_preconditioner(self)

    def ilu0_factor(self) -> ILU0Factorization:
        return sparse_ilu0_factor(self)

    def ilut_factor(
        self,
        *,
        drop_tolerance: float | None = None,
        max_fill: int | None = None,
        config: ILUTConfig | None = None,
    ) -> ILUTFactorization:
        return sparse_ilut_factor(
            self,
            drop_tolerance=drop_tolerance,
            max_fill=max_fill,
            config=config,
        )

    def iluk_factor(
        self,
        *,
        level_of_fill: int | None = None,
        config: ILUKConfig | None = None,
    ) -> ILUKFactorization:
        return sparse_iluk_factor(self, level_of_fill=level_of_fill, config=config)

    def ic0_factor(self) -> IC0Factorization:
        return sparse_ic0_factor(self)

    def ildl0_factor(self) -> ILDL0Factorization:
        return sparse_ildl0_factor(self)

    def lu_factor(self) -> SparseLUFactorization:
        return sparse_lu_factor(self)

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, (CsrMatrix, CscMatrix, CooMatrix)):
            return sparse_matmat_sparse(self, other)
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
    if isinstance(matrix, CscMatrix):
        return matrix.to_csr()
    if isinstance(matrix, CooMatrix):
        return matrix.to_csr()
    return CsrMatrix.from_scipy(matrix)


def sparse_matvec(matrix: Any, vector: Any) -> np.ndarray:
    if isinstance(matrix, CscMatrix):
        return sparse_matvec_csc(matrix, vector)
    csr = _coerce_csr_matrix(matrix)
    return _sparse_matvec_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_vector(vector, dtype=csr.data.dtype),
    )


def sparse_matvec_csc(matrix: Any, vector: Any) -> np.ndarray:
    csc = matrix if isinstance(matrix, CscMatrix) else CscMatrix.from_scipy(matrix)
    return _sparse_matvec_csc_raw(
        csc.nrows,
        csc.ncols,
        csc.indptr,
        csc.indices,
        csc.data,
        _normalize_vector(vector, dtype=csc.data.dtype),
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


def sparse_csr_to_csc(matrix: Any) -> CscMatrix:
    csr = _coerce_csr_matrix(matrix)
    return CscMatrix.from_components(
        *_sparse_csr_to_csc_raw(csr.nrows, csr.ncols, csr.indptr, csr.indices, csr.data)
    )


def sparse_csc_to_csr(matrix: Any) -> CsrMatrix:
    csc = matrix if isinstance(matrix, CscMatrix) else CscMatrix.from_scipy(matrix)
    return CsrMatrix.from_components(
        *_sparse_csc_to_csr_raw(csc.nrows, csc.ncols, csc.indptr, csc.indices, csc.data)
    )


def sparse_coo_to_csr(matrix: Any) -> CsrMatrix:
    coo = matrix if isinstance(matrix, CooMatrix) else CooMatrix.from_scipy(matrix)
    return CsrMatrix.from_components(
        *_sparse_coo_to_csr_raw(
            coo.nrows,
            coo.ncols,
            coo.row_indices,
            coo.col_indices,
            coo.data,
        )
    )


def sparse_matmat_sparse(left: Any, right: Any) -> CsrMatrix:
    left_csr = _coerce_csr_matrix(left)
    right_csr = _coerce_csr_matrix(right)
    return CsrMatrix.from_components(
        *_sparse_matmat_sparse_raw(
            left_csr.nrows,
            left_csr.ncols,
            left_csr.indptr,
            left_csr.indices,
            left_csr.data,
            right_csr.nrows,
            right_csr.ncols,
            right_csr.indptr,
            right_csr.indices,
            right_csr.data,
        )
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


def sparse_gauss_seidel_solve(
    matrix: Any,
    rhs: Any,
    *,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> np.ndarray:
    csr = _coerce_csr_matrix(matrix)
    return _sparse_gauss_seidel_solve_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_rhs(rhs, dtype=csr.data.dtype),
        tolerance,
        max_iterations,
    )


def sparse_conjugate_gradient_solve(
    matrix: Any,
    rhs: Any,
    *,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> np.ndarray:
    csr = _coerce_csr_matrix(matrix)
    return _sparse_conjugate_gradient_solve_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_rhs(rhs, dtype=csr.data.dtype),
        tolerance,
        max_iterations,
    )


def sparse_pcg_ic0_solve(
    matrix: Any,
    rhs: Any,
    *,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> np.ndarray:
    csr = _coerce_csr_matrix(matrix)
    return _sparse_pcg_ic0_solve_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_rhs(rhs, dtype=csr.data.dtype),
        tolerance,
        max_iterations,
    )


def sparse_bicgstab_solve(
    matrix: Any,
    rhs: Any,
    *,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> np.ndarray:
    csr = _coerce_csr_matrix(matrix)
    return _sparse_bicgstab_solve_raw(
        csr.nrows,
        csr.ncols,
        csr.indptr,
        csr.indices,
        csr.data,
        _normalize_rhs(rhs, dtype=csr.data.dtype),
        tolerance,
        max_iterations,
    )


def sparse_jacobi_preconditioner(matrix: Any) -> JacobiPreconditioner:
    csr = _coerce_csr_matrix(matrix)
    return JacobiPreconditioner(
        _sparse_jacobi_preconditioner_raw(csr.nrows, csr.ncols, csr.indptr, csr.indices, csr.data)
    )


def sparse_ilu0_factor(matrix: Any) -> ILU0Factorization:
    csr = _coerce_csr_matrix(matrix)
    return ILU0Factorization(
        csr, _sparse_ilu0_factor_raw(csr.nrows, csr.ncols, csr.indptr, csr.indices, csr.data)
    )


def sparse_ilut_factor(
    matrix: Any,
    *,
    drop_tolerance: float | None = None,
    max_fill: int | None = None,
    config: ILUTConfig | None = None,
) -> ILUTFactorization:
    csr = _coerce_csr_matrix(matrix)
    resolved = _coerce_ilut_config(
        csr, config=config, drop_tolerance=drop_tolerance, max_fill=max_fill
    )
    return ILUTFactorization(
        csr,
        _sparse_ilut_factor_raw(
            csr.nrows,
            csr.ncols,
            csr.indptr,
            csr.indices,
            csr.data,
            float(resolved.drop_tolerance),
            int(resolved.max_fill),
        ),
    )


def sparse_iluk_factor(
    matrix: Any,
    *,
    level_of_fill: int | None = None,
    config: ILUKConfig | None = None,
) -> ILUKFactorization:
    csr = _coerce_csr_matrix(matrix)
    resolved = _coerce_iluk_config(config=config, level_of_fill=level_of_fill)
    return ILUKFactorization(
        csr,
        _sparse_iluk_factor_raw(
            csr.nrows,
            csr.ncols,
            csr.indptr,
            csr.indices,
            csr.data,
            int(resolved.level_of_fill),
        ),
    )


def sparse_ic0_factor(matrix: Any) -> IC0Factorization:
    csr = _coerce_csr_matrix(matrix)
    return IC0Factorization(
        csr, _sparse_ic0_factor_raw(csr.nrows, csr.ncols, csr.indptr, csr.indices, csr.data)
    )


def sparse_ildl0_factor(matrix: Any) -> ILDL0Factorization:
    csr = _coerce_csr_matrix(matrix)
    return ILDL0Factorization(
        csr, _sparse_ildl0_factor_raw(csr.nrows, csr.ncols, csr.indptr, csr.indices, csr.data)
    )


def sparse_lu_factor(matrix: Any) -> SparseLUFactorization:
    csr = _coerce_csr_matrix(matrix)
    return SparseLUFactorization(
        csr, _sparse_lu_factor_raw(csr.nrows, csr.ncols, csr.indptr, csr.indices, csr.data)
    )


def sparse_lu_solve(matrix: Any, rhs: Any) -> np.ndarray:
    return sparse_lu_factor(matrix).solve(rhs)


__all__ = [
    "CooMatrix",
    "CscMatrix",
    "CsrMatrix",
    "IC0Factorization",
    "ILDL0Factorization",
    "ILU0Factorization",
    "ILUKConfig",
    "ILUKFactorization",
    "ILUTConfig",
    "ILUTFactorization",
    "JacobiPreconditioner",
    "SparseLUFactorization",
    "sparse_bicgstab_solve",
    "sparse_conjugate_gradient_solve",
    "sparse_ic0_factor",
    "sparse_ildl0_factor",
    "sparse_ilu0_factor",
    "sparse_iluk_factor",
    "sparse_ilut_factor",
    "sparse_coo_to_csr",
    "sparse_csc_to_csr",
    "sparse_csr_to_csc",
    "sparse_gauss_seidel_solve",
    "sparse_jacobi_preconditioner",
    "sparse_lu_factor",
    "sparse_lu_solve",
    "sparse_matmat_sparse",
    "sparse_matvec",
    "sparse_matvec_csc",
    "sparse_matmat_dense",
    "sparse_transpose",
    "sparse_jacobi_solve",
    "sparse_pcg_ic0_solve",
    "sparse_pcg_solve",
]
