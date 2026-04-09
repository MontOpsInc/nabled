"""Reusable Python workspace objects for pynabled hot paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from pynabled._pynabled import (
    _MatrixFunctionWorkspace as _RawMatrixFunctionWorkspace,
)
from pynabled._pynabled import (
    _PairwiseCosineWorkspace as _RawPairwiseCosineWorkspace,
)
from pynabled._pynabled import (
    _SylvesterWorkspace as _RawSylvesterWorkspace,
)

_REAL_WORKSPACE_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_NUMERIC_WORKSPACE_DTYPES = (*_REAL_WORKSPACE_DTYPES, np.dtype(np.complex128))


def _normalize_workspace_dtype(
    name: str,
    dtype: Any,
    allowed: tuple[np.dtype[Any], ...],
) -> np.dtype[Any]:
    resolved = np.dtype(dtype)
    if resolved not in allowed:
        allowed_names = " or ".join(candidate.name for candidate in allowed)
        raise TypeError(f"{name} must be {allowed_names}")
    return resolved


class PairwiseCosineWorkspace:
    """Reusable workspace for repeated pairwise cosine similarity/distance calls."""

    __slots__ = ("_dtype", "_raw")

    def __init__(self, dtype: Any = np.float64):
        self._dtype = _normalize_workspace_dtype("dtype", dtype, _REAL_WORKSPACE_DTYPES)
        self._raw = _RawPairwiseCosineWorkspace(self._dtype.name)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def similarity(self, left, right, *, out=None):
        if out is None:
            return self._raw.similarity(left, right)
        self._raw.similarity_into(left, right, out)
        return out

    def distance(self, left, right, *, out=None):
        if out is None:
            return self._raw.distance(left, right)
        self._raw.distance_into(left, right, out)
        return out

    def __repr__(self) -> str:  # pragma: no cover - trivial repr
        return f"PairwiseCosineWorkspace(dtype={self._dtype.name!r})"


class MatrixFunctionWorkspace:
    """Reusable workspace for repeated matrix-function kernels."""

    __slots__ = ("_dtype", "_raw")

    def __init__(self, dtype: Any = np.float64):
        self._dtype = _normalize_workspace_dtype("dtype", dtype, _NUMERIC_WORKSPACE_DTYPES)
        self._raw = _RawMatrixFunctionWorkspace(self._dtype.name)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def exp(self, matrix, max_terms=None, tolerance=None, *, out=None):
        if out is None:
            return self._raw.exp(matrix, max_terms=max_terms, tolerance=tolerance)
        self._raw.exp_into(matrix, out, max_terms=max_terms, tolerance=tolerance)
        return out

    def log_taylor(self, matrix, max_terms=None, tolerance=None, *, out=None):
        if self._dtype == np.dtype(np.complex128):
            raise TypeError("matrix_log_taylor workspace must use dtype float32 or float64")
        if out is None:
            return self._raw.log_taylor(matrix, max_terms=max_terms, tolerance=tolerance)
        self._raw.log_taylor_into(matrix, out, max_terms=max_terms, tolerance=tolerance)
        return out

    def log_eigen(self, matrix, *, out=None):
        if out is None:
            return self._raw.log_eigen(matrix)
        self._raw.log_eigen_into(matrix, out)
        return out

    def log_svd(self, matrix, *, out=None):
        if out is None:
            return self._raw.log_svd(matrix)
        self._raw.log_svd_into(matrix, out)
        return out

    def power(self, matrix, power, *, out=None):
        if out is None:
            return self._raw.power(matrix, power)
        self._raw.power_into(matrix, power, out)
        return out

    def sign(self, matrix, *, out=None):
        if out is None:
            return self._raw.sign(matrix)
        self._raw.sign_into(matrix, out)
        return out

    def __repr__(self) -> str:  # pragma: no cover - trivial repr
        return f"MatrixFunctionWorkspace(dtype={self._dtype.name!r})"


class SylvesterWorkspace:
    """Reusable workspace for Sylvester and Lyapunov solves."""

    __slots__ = ("_dtype", "_raw")

    def __init__(self, dtype: Any = np.float64):
        self._dtype = _normalize_workspace_dtype("dtype", dtype, _NUMERIC_WORKSPACE_DTYPES)
        self._raw = _RawSylvesterWorkspace(self._dtype.name)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def solve(self, matrix_a, matrix_b, matrix_c, *, out=None):
        if out is None:
            return self._raw.solve(matrix_a, matrix_b, matrix_c)
        self._raw.solve_into(matrix_a, matrix_b, matrix_c, out)
        return out

    def lyapunov(self, a, q, *, out=None):
        if out is None:
            return self._raw.lyapunov(a, q)
        self._raw.lyapunov_into(a, q, out)
        return out

    def __repr__(self) -> str:  # pragma: no cover - trivial repr
        return f"SylvesterWorkspace(dtype={self._dtype.name!r})"


__all__ = [
    "MatrixFunctionWorkspace",
    "PairwiseCosineWorkspace",
    "SylvesterWorkspace",
]
