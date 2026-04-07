"""PyArrow bridge for nabled/ndarrow workflows.

Requires pynabled built with the arrow feature: ``pip install pynabled[arrow]``
and ``maturin develop --features arrow`` (or equivalent).
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from .config import (
    AdamConfig,
    BFGSConfig,
    GradientDescentConfig,
    IterativeConfig,
    JacobianConfig,
    LineSearchConfig,
    MomentumConfig,
    ProjectedGradientConfig,
    RMSPropConfig,
)
from .results import (
    CholeskyResult,
    EigenResult,
    GeneralizedEigenResult,
    LogDetResult,
    LuResult,
    NonsymmetricBiEigenResult,
    NonsymmetricEigenResult,
    PcaResult,
    PolarResult,
    QrResult,
    RegressionResult,
    SchurResult,
    SvdResult,
)
from .sparse import (
    CscMatrix,
    CsrMatrix,
    IC0Factorization,
    ILDL0Factorization,
    ILU0Factorization,
    ILUKConfig,
    ILUKFactorization,
    ILUTConfig,
    ILUTFactorization,
    JacobiPreconditioner,
    SparseLUFactorization,
)

try:
    import pynabled._pynabled as _raw
except ImportError as e:
    raise ImportError(
        "pynabled arrow support not available. "
        "Install with: pip install pynabled[arrow] and build with --features arrow"
    ) from e


if not hasattr(_raw, "arrow_dot"):
    raise ImportError(
        "pynabled arrow support not available. "
        "Install with: pip install pynabled[arrow] and build with --features arrow"
    )


_COMPLEX_EXTENSION_NAMES = {"ndarrow.complex64"}
_COMPLEX_STORAGE_TYPE = pa.list_(pa.field("item", pa.float64(), nullable=False), 2)
_CSR_EXTENSION_NAME = "ndarrow.csr_matrix"
_CSR_BATCH_EXTENSION_NAME = "ndarrow.csr_matrix_batch"
_VARIABLE_SHAPE_TENSOR_EXTENSION_NAME = "arrow.variable_shape_tensor"
_UINT32_MAX = np.iinfo(np.uint32).max
_INT32_MAX = np.iinfo(np.int32).max


def _list_value_type(type_):
    return type_.value_field.type if hasattr(type_, "value_field") else type_.value_type


class NdarrowCsrMatrixType(pa.ExtensionType):
    def __init__(self, value_type, ncols: int):
        self._value_type = value_type
        self._ncols = int(ncols)
        super().__init__(
            pa.struct(
                [
                    pa.field(
                        "indices",
                        pa.list_(pa.field("item", pa.uint32(), nullable=False)),
                        nullable=False,
                    ),
                    pa.field(
                        "values",
                        pa.list_(pa.field("item", value_type, nullable=False)),
                        nullable=False,
                    ),
                ]
            ),
            _CSR_EXTENSION_NAME,
        )

    def __arrow_ext_serialize__(self):
        return json.dumps({"ncols": self._ncols}).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        metadata = json.loads(serialized.decode()) if serialized else {}
        value_type = _list_value_type(storage_type.field("values").type)
        return cls(value_type, metadata["ncols"])

    def __reduce__(self):
        return NdarrowCsrMatrixType, (self._value_type, self._ncols)


class NdarrowCsrMatrixBatchType(pa.ExtensionType):
    def __init__(self, value_type):
        self._value_type = value_type
        super().__init__(
            pa.struct(
                [
                    pa.field(
                        "shape",
                        pa.list_(pa.field("item", pa.int32(), nullable=False), 2),
                        nullable=False,
                    ),
                    pa.field(
                        "row_ptrs",
                        pa.list_(pa.field("item", pa.int32(), nullable=False)),
                        nullable=False,
                    ),
                    pa.field(
                        "col_indices",
                        pa.list_(pa.field("item", pa.uint32(), nullable=False)),
                        nullable=False,
                    ),
                    pa.field(
                        "values",
                        pa.list_(pa.field("item", value_type, nullable=False)),
                        nullable=False,
                    ),
                ]
            ),
            _CSR_BATCH_EXTENSION_NAME,
        )

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        del serialized
        value_type = _list_value_type(storage_type.field("values").type)
        return cls(value_type)

    def __reduce__(self):
        return NdarrowCsrMatrixBatchType, (self._value_type,)


class ArrowVariableShapeTensorType(pa.ExtensionType):
    def __init__(self, value_type, dimensions: int, uniform_shape):
        self._value_type = value_type
        self._dimensions = int(dimensions)
        self._uniform_shape = None if uniform_shape is None else list(uniform_shape)
        super().__init__(
            pa.struct(
                [
                    pa.field(
                        "data",
                        pa.list_(pa.field("item", value_type, nullable=False)),
                        nullable=False,
                    ),
                    pa.field(
                        "shape",
                        pa.list_(
                            pa.field("item", pa.int32(), nullable=False),
                            self._dimensions,
                        ),
                        nullable=False,
                    ),
                ]
            ),
            _VARIABLE_SHAPE_TENSOR_EXTENSION_NAME,
        )

    def __arrow_ext_serialize__(self):
        return json.dumps(
            {
                "dim_names": None,
                "permutations": None,
                "uniform_shape": self._uniform_shape,
            }
        ).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        metadata = json.loads(serialized.decode()) if serialized else {}
        data_type = _list_value_type(storage_type.field("data").type)
        shape_type = storage_type.field("shape").type
        dimensions = shape_type.list_size if hasattr(shape_type, "list_size") else len(shape_type)
        return cls(data_type, dimensions, metadata.get("uniform_shape"))

    def __reduce__(self):
        return ArrowVariableShapeTensorType, (
            self._value_type,
            self._dimensions,
            self._uniform_shape,
        )


def _try_register_extension(extension_type):
    try:
        pa.register_extension_type(extension_type)
    except Exception:
        pass


_try_register_extension(NdarrowCsrMatrixType(pa.float64(), 0))
_try_register_extension(NdarrowCsrMatrixBatchType(pa.float64()))
_try_register_extension(ArrowVariableShapeTensorType(pa.float64(), 1, None))


def _resolve_ilut_config(*, config: ILUTConfig | None, drop_tolerance, max_fill) -> ILUTConfig:
    if config is not None and (drop_tolerance is not None or max_fill is not None):
        raise TypeError("pass either config=... or explicit drop_tolerance/max_fill, not both")
    if config is not None:
        return config
    base = ILUTConfig.balanced()
    return ILUTConfig(
        drop_tolerance=base.drop_tolerance if drop_tolerance is None else float(drop_tolerance),
        max_fill=base.max_fill if max_fill is None else int(max_fill),
    )


def _resolve_iluk_config(*, config: ILUKConfig | None, level_of_fill) -> ILUKConfig:
    if config is not None and level_of_fill is not None:
        raise TypeError("pass either config=... or explicit level_of_fill, not both")
    if config is not None:
        return config
    return ILUKConfig.balanced() if level_of_fill is None else ILUKConfig(int(level_of_fill))


def _csr_matrix_batch_field(name, storage_type):
    return pa.field(
        name,
        storage_type,
        nullable=False,
        metadata={"ARROW:extension:name": _CSR_BATCH_EXTENSION_NAME},
    )


def _arrow_field(array, name):
    if isinstance(array, pa.ExtensionArray) and _extension_name(array.type) in _COMPLEX_EXTENSION_NAMES:
        return _complex_vector_field(name)
    if _is_extension_array_named(array, _CSR_BATCH_EXTENSION_NAME):
        return _csr_matrix_batch_field(name, array.storage.type)
    return pa.field(name, array.type, nullable=False)


def _extension_array(field, storage):
    return pa.ExtensionArray.from_storage(field.type, storage)


def _extension_name(value):
    type_ = value.type if hasattr(value, "type") else value
    return getattr(type_, "extension_name", None)


def _is_extension_type(type_):
    return isinstance(type_, pa.ExtensionType)


def _is_complex_scalar_type(type_):
    return _is_extension_type(type_) and _extension_name(type_) in _COMPLEX_EXTENSION_NAMES


def _field_metadata_value(field, key):
    metadata = field.metadata or {}
    if key in metadata:
        return metadata[key]
    return metadata.get(key.encode())


def _complex_vector_field(name):
    return pa.field(
        name,
        _COMPLEX_STORAGE_TYPE,
        nullable=False,
        metadata={"ARROW:extension:name": "ndarrow.complex64"},
    )


def _complex_vector_storage(array):
    if isinstance(array, pa.ExtensionArray) and _extension_name(array.type) in _COMPLEX_EXTENSION_NAMES:
        return array.storage
    return None


def _complex_matrix_storage(array):
    if not isinstance(array, pa.FixedSizeListArray):
        return None
    value_field = array.type.value_field
    if (
        value_field.type == _COMPLEX_STORAGE_TYPE
        and _field_metadata_value(value_field, "ARROW:extension:name") in ("ndarrow.complex64", b"ndarrow.complex64")
    ):
        return array
    if _is_extension_type(value_field.type) and _extension_name(value_field.type) in _COMPLEX_EXTENSION_NAMES:
        return pa.FixedSizeListArray.from_arrays(
            array.values.storage,
            type=pa.list_(_complex_vector_field("item"), array.type.list_size),
        )
    return None


def _is_complex_vector(array):
    return _complex_vector_storage(array) is not None


def _is_complex_matrix(array):
    return _complex_matrix_storage(array) is not None


def _is_complex_array(array):
    return _is_complex_vector(array) or _is_complex_matrix(array)


def _require_complex(name, **arrays):
    wrong = [label for label, array in arrays.items() if not _is_complex_array(array)]
    if wrong:
        joined = ", ".join(wrong)
        raise TypeError(f"{name} requires ndarrow.complex64 Arrow carriers for: {joined}")


def _require_real(name, **arrays):
    wrong = [label for label, array in arrays.items() if _is_complex_array(array)]
    if wrong:
        joined = ", ".join(wrong)
        raise TypeError(f"{name} does not currently admit ndarrow.complex64 Arrow carriers: {joined}")


def _complex_mode(name, **arrays):
    flags = {label: _is_complex_array(array) for label, array in arrays.items()}
    if any(flags.values()) and not all(flags.values()):
        joined = ", ".join(arrays)
        raise TypeError(
            f"{name} requires {joined} to all be real Arrow carriers or all ndarrow.complex64 carriers"
        )
    return any(flags.values())


def _is_extension_array_named(array, name: str):
    return isinstance(array, pa.ExtensionArray) and _extension_name(array.type) == name


def _require_extension_array(name, value, extension_name: str, label: str):
    if not _is_extension_array_named(value, extension_name):
        raise TypeError(f"{name} requires {label} to be a {extension_name} PyArrow extension array")


def _require_csr_matrix_array(name, matrix):
    _require_extension_array(name, matrix, _CSR_EXTENSION_NAME, "matrix")


def _require_csr_matrix_batch_array(name, matrices):
    _require_extension_array(name, matrices, _CSR_BATCH_EXTENSION_NAME, "matrices")


def _require_variable_shape_tensor_array(name, value, label: str):
    _require_extension_array(name, value, _VARIABLE_SHAPE_TENSOR_EXTENSION_NAME, label)


def _arrow_real_type_for_dtype(dtype) -> pa.DataType:
    resolved = np.dtype(dtype)
    if resolved == np.dtype(np.float32):
        return pa.float32()
    if resolved == np.dtype(np.float64):
        return pa.float64()
    raise TypeError("dtype must be float32 or float64")


def _numpy_real_dtype_for_arrow(type_) -> np.dtype[np.generic]:
    if pa.types.is_float32(type_):
        return np.dtype(np.float32)
    if pa.types.is_float64(type_):
        return np.dtype(np.float64)
    raise TypeError("expected float32 or float64 Arrow values")


def _flatten_csr_lists(csr: CsrMatrix):
    if np.any(csr.indices > _UINT32_MAX):
        raise ValueError("csr column indices exceed uint32 limits required by ndarrow.csr_matrix")
    index_rows = []
    value_rows = []
    for row in range(csr.nrows):
        start = int(csr.indptr[row])
        stop = int(csr.indptr[row + 1])
        index_rows.append(csr.indices[start:stop].astype(np.uint32, copy=False).tolist())
        value_rows.append(csr.data[start:stop].tolist())
    return index_rows, value_rows


def arrow_csr_matrix_array(
    matrix,
    *,
    copy: bool = False,
    dtype=None,
    index_dtype=None,
):
    if _is_extension_array_named(matrix, _CSR_EXTENSION_NAME):
        if dtype is None and index_dtype is None and not copy:
            return matrix
        matrix = arrow_csr_matrix_from_array(matrix)
    csr = CsrMatrix.from_scipy(matrix, copy=copy, dtype=dtype, index_dtype=index_dtype)
    index_rows, value_rows = _flatten_csr_lists(csr)
    value_type = _arrow_real_type_for_dtype(csr.dtype)
    storage = pa.StructArray.from_arrays(
        [
            pa.array(index_rows, type=pa.list_(pa.field("item", pa.uint32(), nullable=False))),
            pa.array(value_rows, type=pa.list_(pa.field("item", value_type, nullable=False))),
        ],
        fields=[
            pa.field(
                "indices",
                pa.list_(pa.field("item", pa.uint32(), nullable=False)),
                nullable=False,
            ),
            pa.field(
                "values",
                pa.list_(pa.field("item", value_type, nullable=False)),
                nullable=False,
            ),
        ],
    )
    return pa.ExtensionArray.from_storage(NdarrowCsrMatrixType(value_type, csr.ncols), storage)


def arrow_csr_matrix_from_array(matrix) -> CsrMatrix:
    _require_csr_matrix_array("arrow_csr_matrix_from_array", matrix)
    storage = matrix.storage
    index_rows = storage.field("indices").to_pylist()
    value_rows = storage.field("values").to_pylist()
    dtype = _numpy_real_dtype_for_arrow(_list_value_type(storage.field("values").type))
    ncols = int(matrix.type._ncols if hasattr(matrix.type, "_ncols") else json.loads(matrix.type.__arrow_ext_serialize__().decode())["ncols"])
    indptr = [0]
    flat_indices = []
    flat_values = []
    for row_indices, row_values in zip(index_rows, value_rows, strict=True):
        if len(row_indices) != len(row_values):
            raise ValueError("ndarrow.csr_matrix storage has mismatched row lengths")
        flat_indices.extend(row_indices)
        flat_values.extend(row_values)
        indptr.append(len(flat_indices))
    return CsrMatrix.from_components(
        (len(index_rows), ncols),
        np.asarray(indptr, dtype=np.int32),
        np.asarray(flat_indices, dtype=np.int32),
        np.asarray(flat_values, dtype=dtype),
    )


def arrow_csr_matrix_batch_array(matrices, *, copy: bool = False, dtype=None, index_dtype=None):
    rows = [CsrMatrix.from_scipy(matrix, copy=copy, dtype=dtype, index_dtype=index_dtype) for matrix in matrices]
    if not rows:
        raise ValueError("arrow_csr_matrix_batch_array requires at least one matrix")
    resolved_dtype = rows[0].dtype
    if any(row.dtype != resolved_dtype for row in rows[1:]):
        raise TypeError("all matrices in a csr_matrix_batch must share dtype float32 or float64")
    shapes = []
    row_ptrs = []
    col_indices = []
    values = []
    for row in rows:
        if np.any(row.indptr > _INT32_MAX):
            raise ValueError("csr row pointers exceed int32 limits required by ndarrow.csr_matrix_batch")
        if np.any(row.indices > _UINT32_MAX):
            raise ValueError("csr column indices exceed uint32 limits required by ndarrow.csr_matrix_batch")
        shapes.append([row.nrows, row.ncols])
        row_ptrs.append(row.indptr.astype(np.int32, copy=False).tolist())
        col_indices.append(row.indices.astype(np.uint32, copy=False).tolist())
        values.append(row.data.tolist())
    value_type = _arrow_real_type_for_dtype(resolved_dtype)
    storage = pa.StructArray.from_arrays(
        [
            pa.array(
                shapes,
                type=pa.list_(pa.field("item", pa.int32(), nullable=False), 2),
            ),
            pa.array(
                row_ptrs,
                type=pa.list_(pa.field("item", pa.int32(), nullable=False)),
            ),
            pa.array(
                col_indices,
                type=pa.list_(pa.field("item", pa.uint32(), nullable=False)),
            ),
            pa.array(
                values,
                type=pa.list_(pa.field("item", value_type, nullable=False)),
            ),
        ],
        fields=[
            pa.field(
                "shape",
                pa.list_(pa.field("item", pa.int32(), nullable=False), 2),
                nullable=False,
            ),
            pa.field(
                "row_ptrs",
                pa.list_(pa.field("item", pa.int32(), nullable=False)),
                nullable=False,
            ),
            pa.field(
                "col_indices",
                pa.list_(pa.field("item", pa.uint32(), nullable=False)),
                nullable=False,
            ),
            pa.field(
                "values",
                pa.list_(pa.field("item", value_type, nullable=False)),
                nullable=False,
            ),
        ],
    )
    return pa.ExtensionArray.from_storage(NdarrowCsrMatrixBatchType(value_type), storage)


def arrow_csr_matrix_batch_rows(matrices) -> list[CsrMatrix]:
    _require_csr_matrix_batch_array("arrow_csr_matrix_batch_rows", matrices)
    storage = matrices.storage
    shapes = storage.field("shape").to_pylist()
    row_ptrs = storage.field("row_ptrs").to_pylist()
    col_indices = storage.field("col_indices").to_pylist()
    values = storage.field("values").to_pylist()
    dtype = _numpy_real_dtype_for_arrow(_list_value_type(storage.field("values").type))
    rows = []
    for shape, indptr, indices, data in zip(shapes, row_ptrs, col_indices, values, strict=True):
        rows.append(
            CsrMatrix.from_components(
                tuple(shape),
                np.asarray(indptr, dtype=np.int32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(data, dtype=dtype),
            )
        )
    return rows


def arrow_variable_shape_tensor_array(rows, *, uniform_shape=None, dtype=None):
    numpy_rows = [np.asarray(row, dtype=dtype) if dtype is not None else np.asarray(row) for row in rows]
    if not numpy_rows:
        raise ValueError("arrow_variable_shape_tensor_array requires at least one tensor")
    rank = numpy_rows[0].ndim
    if rank == 0:
        raise ValueError("arrow_variable_shape_tensor_array requires tensors with rank >= 1")
    resolved_dtype = numpy_rows[0].dtype
    if resolved_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("arrow_variable_shape_tensor_array currently supports float32 or float64")
    if any(row.ndim != rank for row in numpy_rows[1:]):
        raise ValueError("all tensors in a variable_shape_tensor batch must share rank")
    if any(row.dtype != resolved_dtype for row in numpy_rows[1:]):
        raise TypeError("all tensors in a variable_shape_tensor batch must share dtype")
    normalized_uniform = None if uniform_shape is None else [None if item is None else int(item) for item in uniform_shape]
    if normalized_uniform is not None and len(normalized_uniform) != rank:
        raise ValueError("uniform_shape length must match tensor rank")
    packed_data = [row.reshape(-1).tolist() for row in numpy_rows]
    packed_shapes = [list(map(int, row.shape)) for row in numpy_rows]
    value_type = _arrow_real_type_for_dtype(resolved_dtype)
    storage = pa.StructArray.from_arrays(
        [
            pa.array(
                packed_data,
                type=pa.list_(pa.field("item", value_type, nullable=False)),
            ),
            pa.array(
                packed_shapes,
                type=pa.list_(pa.field("item", pa.int32(), nullable=False), rank),
            ),
        ],
        fields=[
            pa.field(
                "data",
                pa.list_(pa.field("item", value_type, nullable=False)),
                nullable=False,
            ),
            pa.field(
                "shape",
                pa.list_(pa.field("item", pa.int32(), nullable=False), rank),
                nullable=False,
            ),
        ],
    )
    return pa.ExtensionArray.from_storage(
        ArrowVariableShapeTensorType(value_type, rank, normalized_uniform),
        storage,
    )


def arrow_variable_shape_tensor_rows(value) -> list[np.ndarray]:
    _require_variable_shape_tensor_array("arrow_variable_shape_tensor_rows", value, "value")
    storage = value.storage
    dtype = _numpy_real_dtype_for_arrow(_list_value_type(storage.field("data").type))
    rows = []
    for packed, shape in zip(storage.field("data").to_pylist(), storage.field("shape").to_pylist(), strict=True):
        rows.append(np.asarray(packed, dtype=dtype).reshape(tuple(int(dimension) for dimension in shape)))
    return rows


def _resolve_config(config, config_type, **kwargs):
    if config is None:
        return kwargs
    if not isinstance(config, config_type):
        raise TypeError(f"config must be {config_type.__name__} or None")
    conflicts = [name for name, value in kwargs.items() if value is not None]
    if conflicts:
        joined = ", ".join(conflicts)
        raise TypeError(
            f"pass either {config_type.__name__} via config= or explicit keyword arguments, not both: {joined}"
        )
    return {name: getattr(config, name) for name in kwargs}


def _svd_result(raw_result) -> SvdResult:
    u, singular_values, vt = raw_result
    return SvdResult(u=u, singular_values=singular_values, vt=vt)


def _qr_result(raw_result) -> QrResult:
    if len(raw_result) == 3:
        q, r, rank = raw_result
        return QrResult(q=q, r=r, rank=rank)
    q, r, p, rank = raw_result
    return QrResult(q=q, r=r, rank=rank, p=p)


def _lu_result(raw_result) -> LuResult:
    l, u = raw_result
    return LuResult(l=l, u=u)


def _cholesky_result(raw_result) -> CholeskyResult:
    return CholeskyResult(l=raw_result)


def _eigen_result(raw_result) -> EigenResult:
    eigenvalues, eigenvectors = raw_result
    return EigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def _generalized_eigen_result(raw_result) -> GeneralizedEigenResult:
    eigenvalues, eigenvectors = raw_result
    return GeneralizedEigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def _nonsymmetric_eigen_result(raw_result) -> NonsymmetricEigenResult:
    eigenvalues, schur_vectors = raw_result
    return NonsymmetricEigenResult(eigenvalues=eigenvalues, schur_vectors=schur_vectors)


def _nonsymmetric_bi_eigen_result(raw_result) -> NonsymmetricBiEigenResult:
    (
        eigenvalues,
        right_eigenvectors,
        left_eigenvectors,
        balancing_diagonal,
        balanced_matrix,
    ) = raw_result
    return NonsymmetricBiEigenResult(
        eigenvalues=eigenvalues,
        right_eigenvectors=right_eigenvectors,
        left_eigenvectors=left_eigenvectors,
        balancing_diagonal=balancing_diagonal,
        balanced_matrix=balanced_matrix,
    )


def _schur_result(raw_result) -> SchurResult:
    t, q = raw_result
    return SchurResult(q=q, t=t)


def _polar_result(raw_result) -> PolarResult:
    u, p = raw_result
    return PolarResult(u=u, p=p)


def _log_det_result(raw_result) -> LogDetResult:
    sign, ln_abs_det = raw_result
    return LogDetResult(sign=sign, ln_abs_det=float(ln_abs_det))


def _pca_result(raw_result) -> PcaResult:
    components, explained_variance, explained_variance_ratio, mean, scores = raw_result
    return PcaResult(
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean,
        scores=scores,
    )


def _regression_result(raw_result) -> RegressionResult:
    coefficients, fitted_values, residuals, r_squared = raw_result
    return RegressionResult(
        coefficients=coefficients,
        fitted_values=fitted_values,
        residuals=residuals,
        r_squared=r_squared,
    )


def arrow_dot_hermitian(left, right):
    _require_complex("arrow_dot_hermitian", left=left, right=right)
    left_storage = _complex_vector_storage(left)
    right_storage = _complex_vector_storage(right)
    return _raw.arrow_dot_hermitian(
        _complex_vector_field("left"),
        left_storage,
        _complex_vector_field("right"),
        right_storage,
    )


def arrow_l2_norm_complex(vector):
    _require_complex("arrow_l2_norm_complex", vector=vector)
    return _raw.arrow_l2_norm_complex(
        _complex_vector_field("vector"),
        _complex_vector_storage(vector),
    )


def arrow_cosine_similarity_complex(left, right):
    _require_complex("arrow_cosine_similarity_complex", left=left, right=right)
    left_storage = _complex_vector_storage(left)
    right_storage = _complex_vector_storage(right)
    field, storage = _raw.arrow_cosine_similarity_complex(
        _complex_vector_field("left"),
        left_storage,
        _complex_vector_field("right"),
        right_storage,
    )
    return _extension_array(field, storage)


def arrow_batched_dot_hermitian(left, right):
    _require_complex("arrow_batched_dot_hermitian", left=left, right=right)
    field, storage = _raw.arrow_batched_dot_hermitian(
        _complex_matrix_storage(left),
        _complex_matrix_storage(right),
    )
    return _extension_array(field, storage)


def arrow_batched_l2_norm_complex(rows):
    _require_complex("arrow_batched_l2_norm_complex", rows=rows)
    return _raw.arrow_batched_l2_norm_complex(_complex_matrix_storage(rows))


def arrow_batched_cosine_similarity_complex(left, right):
    _require_complex("arrow_batched_cosine_similarity_complex", left=left, right=right)
    field, storage = _raw.arrow_batched_cosine_similarity_complex(
        _complex_matrix_storage(left),
        _complex_matrix_storage(right),
    )
    return _extension_array(field, storage)


def arrow_batched_normalize_complex(rows):
    _require_complex("arrow_batched_normalize_complex", rows=rows)
    return _raw.arrow_batched_normalize_complex(_complex_matrix_storage(rows))


def arrow_matvec_complex(matrix, vector):
    _require_complex("arrow_matvec_complex", matrix=matrix, vector=vector)
    matrix_storage = _complex_matrix_storage(matrix)
    vector_storage = _complex_vector_storage(vector)
    field, storage = _raw.arrow_matvec_complex(
        matrix_storage,
        _complex_vector_field("vector"),
        vector_storage,
    )
    return _extension_array(field, storage)


def arrow_matmat_complex(left, right):
    _require_complex("arrow_matmat_complex", left=left, right=right)
    return _raw.arrow_matmat_complex(_complex_matrix_storage(left), _complex_matrix_storage(right))


def arrow_column_means_complex(matrix):
    _require_complex("arrow_column_means_complex", matrix=matrix)
    field, storage = _raw.arrow_column_means_complex(_complex_matrix_storage(matrix))
    return _extension_array(field, storage)


def arrow_center_columns_complex(matrix):
    _require_complex("arrow_center_columns_complex", matrix=matrix)
    return _raw.arrow_center_columns_complex(_complex_matrix_storage(matrix))


def arrow_covariance_matrix_complex(matrix):
    _require_complex("arrow_covariance_matrix_complex", matrix=matrix)
    return _raw.arrow_covariance_matrix_complex(_complex_matrix_storage(matrix))


def arrow_correlation_matrix_complex(matrix):
    _require_complex("arrow_correlation_matrix_complex", matrix=matrix)
    return _raw.arrow_correlation_matrix_complex(_complex_matrix_storage(matrix))


def arrow_gram_schmidt_complex(matrix):
    _require_complex("arrow_gram_schmidt_complex", matrix=matrix)
    return _raw.arrow_gram_schmidt_complex(_complex_matrix_storage(matrix))


def arrow_solve_lower_complex(matrix, rhs):
    _require_complex("arrow_solve_lower_complex", matrix=matrix, rhs=rhs)
    matrix_storage = _complex_matrix_storage(matrix)
    rhs_storage = _complex_vector_storage(rhs)
    field, storage = _raw.arrow_solve_lower_complex(
        matrix_storage,
        _complex_vector_field("rhs"),
        rhs_storage,
    )
    return _extension_array(field, storage)


def arrow_solve_upper_complex(matrix, rhs):
    _require_complex("arrow_solve_upper_complex", matrix=matrix, rhs=rhs)
    matrix_storage = _complex_matrix_storage(matrix)
    rhs_storage = _complex_vector_storage(rhs)
    field, storage = _raw.arrow_solve_upper_complex(
        matrix_storage,
        _complex_vector_field("rhs"),
        rhs_storage,
    )
    return _extension_array(field, storage)


def arrow_batched_matmat(left, right):
    field, storage = _raw.arrow_batched_matmat(
        _arrow_field(left, "left"),
        left,
        _arrow_field(right, "right"),
        right,
    )
    return _extension_array(field, storage)


def arrow_batched_matmat_broadcast_right(left, right):
    field, storage = _raw.arrow_batched_matmat_broadcast_right(_arrow_field(left, "left"), left, right)
    return _extension_array(field, storage)


def arrow_batched_matmat_broadcast_left(left, right):
    field, storage = _raw.arrow_batched_matmat_broadcast_left(left, _arrow_field(right, "right"), right)
    return _extension_array(field, storage)


def arrow_batched_qr(matrices, rank_tolerance=None, max_iterations=None) -> list[QrResult]:
    return [
        _qr_result(result)
        for result in _raw.arrow_batched_qr(
            _arrow_field(matrices, "matrices"),
            matrices,
            rank_tolerance=rank_tolerance,
            max_iterations=max_iterations,
        )
    ]


def arrow_batched_svd(matrices) -> list[SvdResult]:
    return [
        _svd_result(result)
        for result in _raw.arrow_batched_svd(_arrow_field(matrices, "matrices"), matrices)
    ]


def arrow_batched_lu(matrices) -> list[LuResult]:
    return [
        _lu_result(result)
        for result in _raw.arrow_batched_lu(_arrow_field(matrices, "matrices"), matrices)
    ]


def arrow_batched_cholesky(matrices) -> list[CholeskyResult]:
    return [
        _cholesky_result(result)
        for result in _raw.arrow_batched_cholesky(_arrow_field(matrices, "matrices"), matrices)
    ]


def arrow_batched_symmetric_eigen(matrices) -> list[EigenResult]:
    return [
        _eigen_result(result)
        for result in _raw.arrow_batched_symmetric_eigen(_arrow_field(matrices, "matrices"), matrices)
    ]


def arrow_svd_decompose_complex(data) -> SvdResult:
    _require_complex("arrow_svd_decompose_complex", data=data)
    return _svd_result(_raw.arrow_svd_decompose_complex(_complex_matrix_storage(data)))


def arrow_qr_decompose_complex(data) -> QrResult:
    _require_complex("arrow_qr_decompose_complex", data=data)
    return _qr_result(_raw.arrow_qr_decompose_complex(_complex_matrix_storage(data)))


def arrow_lu_solve_complex(matrix, rhs):
    _require_complex("arrow_lu_solve_complex", matrix=matrix, rhs=rhs)
    matrix_storage = _complex_matrix_storage(matrix)
    rhs_storage = _complex_vector_storage(rhs)
    field, storage = _raw.arrow_lu_solve_complex(
        matrix_storage,
        _complex_vector_field("rhs"),
        rhs_storage,
    )
    return _extension_array(field, storage)


def arrow_lu_inverse_complex(matrix):
    _require_complex("arrow_lu_inverse_complex", matrix=matrix)
    return _raw.arrow_lu_inverse_complex(_complex_matrix_storage(matrix))


def arrow_lu_determinant_complex(matrix):
    _require_complex("arrow_lu_determinant_complex", matrix=matrix)
    return _raw.arrow_lu_determinant_complex(_complex_matrix_storage(matrix))


def arrow_cholesky_decompose_complex(matrix) -> CholeskyResult:
    _require_complex("arrow_cholesky_decompose_complex", matrix=matrix)
    return _cholesky_result(_raw.arrow_cholesky_decompose_complex(_complex_matrix_storage(matrix)))


def arrow_cholesky_solve_complex(matrix, rhs):
    _require_complex("arrow_cholesky_solve_complex", matrix=matrix, rhs=rhs)
    matrix_storage = _complex_matrix_storage(matrix)
    rhs_storage = _complex_vector_storage(rhs)
    field, storage = _raw.arrow_cholesky_solve_complex(
        matrix_storage,
        _complex_vector_field("rhs"),
        rhs_storage,
    )
    return _extension_array(field, storage)


def arrow_cholesky_inverse_complex(matrix):
    _require_complex("arrow_cholesky_inverse_complex", matrix=matrix)
    return _raw.arrow_cholesky_inverse_complex(_complex_matrix_storage(matrix))


def arrow_eigen_nonsymmetric_complex(matrix) -> NonsymmetricEigenResult:
    _require_complex("arrow_eigen_nonsymmetric_complex", matrix=matrix)
    return _nonsymmetric_eigen_result(_raw.arrow_eigen_nonsymmetric_complex(_complex_matrix_storage(matrix)))


def arrow_schur_compute_complex(matrix) -> SchurResult:
    _require_complex("arrow_schur_compute_complex", matrix=matrix)
    return _schur_result(_raw.arrow_schur_compute_complex(_complex_matrix_storage(matrix)))


def arrow_polar_compute_complex(matrix) -> PolarResult:
    _require_complex("arrow_polar_compute_complex", matrix=matrix)
    return _polar_result(_raw.arrow_polar_compute_complex(_complex_matrix_storage(matrix)))


def arrow_matrix_exp_complex(matrix, max_terms=None, tolerance=None):
    _require_complex("arrow_matrix_exp_complex", matrix=matrix)
    return _raw.arrow_matrix_exp_complex(
        _complex_matrix_storage(matrix),
        max_terms=max_terms,
        tolerance=tolerance,
    )


def arrow_matrix_exp_eigen_complex(matrix):
    _require_complex("arrow_matrix_exp_eigen_complex", matrix=matrix)
    return _raw.arrow_matrix_exp_eigen_complex(_complex_matrix_storage(matrix))


def arrow_matrix_log_eigen_complex(matrix):
    _require_complex("arrow_matrix_log_eigen_complex", matrix=matrix)
    return _raw.arrow_matrix_log_eigen_complex(_complex_matrix_storage(matrix))


def arrow_matrix_log_svd_complex(matrix):
    _require_complex("arrow_matrix_log_svd_complex", matrix=matrix)
    return _raw.arrow_matrix_log_svd_complex(_complex_matrix_storage(matrix))


def arrow_matrix_power_complex(matrix, power):
    _require_complex("arrow_matrix_power_complex", matrix=matrix)
    return _raw.arrow_matrix_power_complex(_complex_matrix_storage(matrix), power)


def arrow_matrix_sign_complex(matrix):
    _require_complex("arrow_matrix_sign_complex", matrix=matrix)
    return _raw.arrow_matrix_sign_complex(_complex_matrix_storage(matrix))


def arrow_compute_pca_complex(matrix, n_components=None) -> PcaResult:
    _require_complex("arrow_compute_pca_complex", matrix=matrix)
    return _pca_result(
        _raw.arrow_compute_pca_complex(_complex_matrix_storage(matrix), n_components=n_components)
    )


def arrow_pca_transform_complex(matrix, result: PcaResult):
    _require_complex("arrow_pca_transform_complex", matrix=matrix)
    return _raw.arrow_pca_transform_complex(
        _complex_matrix_storage(matrix),
        result.components,
        result.mean,
    )


def arrow_pca_inverse_transform_complex(scores, result: PcaResult):
    _require_complex("arrow_pca_inverse_transform_complex", scores=scores)
    return _raw.arrow_pca_inverse_transform_complex(
        _complex_matrix_storage(scores),
        result.components,
        result.mean,
    )


def arrow_linear_regression_complex(x, y, add_intercept=True) -> RegressionResult:
    _require_complex("arrow_linear_regression_complex", x=x, y=y)
    return _regression_result(
        _raw.arrow_linear_regression_complex(
            _complex_matrix_storage(x),
            _complex_vector_field("y"),
            _complex_vector_storage(y),
            add_intercept=add_intercept,
        )
    )


def arrow_dot(left, right):
    if _complex_mode("arrow_dot", left=left, right=right):
        return arrow_dot_hermitian(left, right)
    return _raw.arrow_dot(left, right)


def arrow_l2_norm(vector):
    if _is_complex_vector(vector):
        return arrow_l2_norm_complex(vector)
    return _raw.arrow_l2_norm(vector)


def arrow_cosine_similarity(left, right):
    if _complex_mode("arrow_cosine_similarity", left=left, right=right):
        return arrow_cosine_similarity_complex(left, right)
    return _raw.arrow_cosine_similarity(left, right)


def arrow_cosine_distance(left, right):
    _require_real("arrow_cosine_distance", left=left, right=right)
    return _raw.arrow_cosine_distance(left, right)


def arrow_pairwise_l2_distance(left, right):
    _require_real("arrow_pairwise_l2_distance", left=left, right=right)
    return _raw.arrow_pairwise_l2_distance(left, right)


def arrow_pairwise_cosine_similarity(left, right):
    _require_real("arrow_pairwise_cosine_similarity", left=left, right=right)
    return _raw.arrow_pairwise_cosine_similarity(left, right)


def arrow_pairwise_cosine_distance(left, right):
    _require_real("arrow_pairwise_cosine_distance", left=left, right=right)
    return _raw.arrow_pairwise_cosine_distance(left, right)


def arrow_batched_dot(left, right):
    if _complex_mode("arrow_batched_dot", left=left, right=right):
        return arrow_batched_dot_hermitian(left, right)
    return _raw.arrow_batched_dot(left, right)


def arrow_batched_l2_norm(rows):
    if _is_complex_matrix(rows):
        return arrow_batched_l2_norm_complex(rows)
    return _raw.arrow_batched_l2_norm(rows)


def arrow_batched_cosine_similarity(left, right):
    if _complex_mode("arrow_batched_cosine_similarity", left=left, right=right):
        return arrow_batched_cosine_similarity_complex(left, right)
    return _raw.arrow_batched_cosine_similarity(left, right)


def arrow_batched_cosine_distance(left, right):
    _require_real("arrow_batched_cosine_distance", left=left, right=right)
    return _raw.arrow_batched_cosine_distance(left, right)


def arrow_batched_normalize(rows):
    if _is_complex_matrix(rows):
        return arrow_batched_normalize_complex(rows)
    return _raw.arrow_batched_normalize(rows)


def arrow_batched_row_matvec(batch_vectors, matrix):
    _require_real("arrow_batched_row_matvec", batch_vectors=batch_vectors, matrix=matrix)
    return _raw.arrow_batched_row_matvec(batch_vectors, matrix)


def arrow_matvec(matrix, vector):
    if _complex_mode("arrow_matvec", matrix=matrix, vector=vector):
        return arrow_matvec_complex(matrix, vector)
    return _raw.arrow_matvec(matrix, vector)


def arrow_matmat(left, right):
    if _complex_mode("arrow_matmat", left=left, right=right):
        return arrow_matmat_complex(left, right)
    return _raw.arrow_matmat(left, right)


def arrow_column_means(matrix):
    if _is_complex_matrix(matrix):
        return arrow_column_means_complex(matrix)
    return _raw.arrow_column_means(matrix)


def arrow_center_columns(matrix):
    if _is_complex_matrix(matrix):
        return arrow_center_columns_complex(matrix)
    return _raw.arrow_center_columns(matrix)


def arrow_covariance_matrix(matrix):
    if _is_complex_matrix(matrix):
        return arrow_covariance_matrix_complex(matrix)
    return _raw.arrow_covariance_matrix(matrix)


def arrow_correlation_matrix(matrix):
    if _is_complex_matrix(matrix):
        return arrow_correlation_matrix_complex(matrix)
    return _raw.arrow_correlation_matrix(matrix)


def arrow_gram_schmidt(matrix):
    if _is_complex_matrix(matrix):
        return arrow_gram_schmidt_complex(matrix)
    return _raw.arrow_gram_schmidt(matrix)


def arrow_gram_schmidt_classic(matrix):
    _require_real("arrow_gram_schmidt_classic", matrix=matrix)
    return _raw.arrow_gram_schmidt_classic(matrix)


def arrow_solve_lower(matrix, rhs):
    if _complex_mode("arrow_solve_lower", matrix=matrix, rhs=rhs):
        return arrow_solve_lower_complex(matrix, rhs)
    return _raw.arrow_solve_lower(matrix, rhs)


def arrow_solve_upper(matrix, rhs):
    if _complex_mode("arrow_solve_upper", matrix=matrix, rhs=rhs):
        return arrow_solve_upper_complex(matrix, rhs)
    return _raw.arrow_solve_upper(matrix, rhs)


def arrow_solve_lower_matrix(matrix, rhs):
    _require_real("arrow_solve_lower_matrix", matrix=matrix, rhs=rhs)
    return _raw.arrow_solve_lower_matrix(matrix, rhs)


def arrow_solve_upper_matrix(matrix, rhs):
    _require_real("arrow_solve_upper_matrix", matrix=matrix, rhs=rhs)
    return _raw.arrow_solve_upper_matrix(matrix, rhs)


def arrow_svd_decompose(data) -> SvdResult:
    if _is_complex_matrix(data):
        return arrow_svd_decompose_complex(data)
    return _svd_result(_raw.arrow_svd_decompose(data))


def arrow_svd_decompose_truncated(data, k) -> SvdResult:
    _require_real("arrow_svd_decompose_truncated", data=data)
    return _svd_result(_raw.arrow_svd_decompose_truncated(data, k))


def arrow_svd_decompose_with_tolerance(data, tolerance) -> SvdResult:
    _require_real("arrow_svd_decompose_with_tolerance", data=data)
    return _svd_result(_raw.arrow_svd_decompose_with_tolerance(data, tolerance))


def arrow_svd_pseudo_inverse(data):
    _require_real("arrow_svd_pseudo_inverse", data=data)
    return _raw.arrow_svd_pseudo_inverse(data)


def arrow_svd_null_space(data, tolerance=None):
    _require_real("arrow_svd_null_space", data=data)
    return _raw.arrow_svd_null_space(data, tolerance)


def arrow_qr_decompose(data, rank_tolerance=None, max_iterations=None) -> QrResult:
    if _is_complex_matrix(data):
        return arrow_qr_decompose_complex(data)
    return _qr_result(
        _raw.arrow_qr_decompose(
            data,
            rank_tolerance=rank_tolerance,
            max_iterations=max_iterations,
        )
    )


def arrow_qr_decompose_reduced(data, rank_tolerance=None, max_iterations=None) -> QrResult:
    _require_real("arrow_qr_decompose_reduced", data=data)
    return _qr_result(
        _raw.arrow_qr_decompose_reduced(
            data,
            rank_tolerance=rank_tolerance,
            max_iterations=max_iterations,
        )
    )


def arrow_qr_decompose_pivoted(data, rank_tolerance=None, max_iterations=None) -> QrResult:
    _require_real("arrow_qr_decompose_pivoted", data=data)
    return _qr_result(
        _raw.arrow_qr_decompose_pivoted(
            data,
            rank_tolerance=rank_tolerance,
            max_iterations=max_iterations,
        )
    )


def arrow_qr_solve_least_squares(a, b, rank_tolerance=None, max_iterations=None):
    _require_real("arrow_qr_solve_least_squares", a=a, b=b)
    return _raw.arrow_qr_solve_least_squares(
        a,
        b,
        rank_tolerance=rank_tolerance,
        max_iterations=max_iterations,
    )


def arrow_lu_decompose(matrix) -> LuResult:
    _require_real("arrow_lu_decompose", matrix=matrix)
    return _lu_result(_raw.arrow_lu_decompose(matrix))


def arrow_lu_solve(matrix, rhs):
    if _complex_mode("arrow_lu_solve", matrix=matrix, rhs=rhs):
        return arrow_lu_solve_complex(matrix, rhs)
    return _raw.arrow_lu_solve(matrix, rhs)


def arrow_lu_inverse(matrix):
    if _is_complex_matrix(matrix):
        return arrow_lu_inverse_complex(matrix)
    return _raw.arrow_lu_inverse(matrix)


def arrow_lu_determinant(matrix):
    if _is_complex_matrix(matrix):
        return arrow_lu_determinant_complex(matrix)
    return _raw.arrow_lu_determinant(matrix)


def arrow_lu_log_determinant(matrix) -> LogDetResult:
    _require_real("arrow_lu_log_determinant", matrix=matrix)
    return _log_det_result(_raw.arrow_lu_log_determinant(matrix))


def arrow_cholesky_decompose(matrix) -> CholeskyResult:
    if _is_complex_matrix(matrix):
        return arrow_cholesky_decompose_complex(matrix)
    return _cholesky_result(_raw.arrow_cholesky_decompose(matrix))


def arrow_cholesky_solve(matrix, rhs):
    if _complex_mode("arrow_cholesky_solve", matrix=matrix, rhs=rhs):
        return arrow_cholesky_solve_complex(matrix, rhs)
    return _raw.arrow_cholesky_solve(matrix, rhs)


def arrow_cholesky_inverse(matrix):
    if _is_complex_matrix(matrix):
        return arrow_cholesky_inverse_complex(matrix)
    return _raw.arrow_cholesky_inverse(matrix)


def arrow_eigen_symmetric(matrix) -> EigenResult:
    _require_real("arrow_eigen_symmetric", matrix=matrix)
    return _eigen_result(_raw.arrow_eigen_symmetric(matrix))


def arrow_eigen_generalized(matrix_a, matrix_b) -> GeneralizedEigenResult:
    _require_real("arrow_eigen_generalized", matrix_a=matrix_a, matrix_b=matrix_b)
    return _generalized_eigen_result(_raw.arrow_eigen_generalized(matrix_a, matrix_b))


def arrow_eigen_nonsymmetric(matrix) -> NonsymmetricEigenResult:
    if _is_complex_matrix(matrix):
        return arrow_eigen_nonsymmetric_complex(matrix)
    return _nonsymmetric_eigen_result(_raw.arrow_eigen_nonsymmetric(matrix))


def arrow_eigen_nonsymmetric_bi(matrix) -> NonsymmetricBiEigenResult:
    _require_real("arrow_eigen_nonsymmetric_bi", matrix=matrix)
    return _nonsymmetric_bi_eigen_result(_raw.arrow_eigen_nonsymmetric_bi(matrix))


def arrow_schur_compute(matrix) -> SchurResult:
    if _is_complex_matrix(matrix):
        return arrow_schur_compute_complex(matrix)
    return _schur_result(_raw.arrow_schur_compute(matrix))


def arrow_polar_compute(matrix) -> PolarResult:
    if _is_complex_matrix(matrix):
        return arrow_polar_compute_complex(matrix)
    return _polar_result(_raw.arrow_polar_compute(matrix))


def arrow_matrix_exp(matrix, max_terms=None, tolerance=None):
    if _is_complex_matrix(matrix):
        return arrow_matrix_exp_complex(matrix, max_terms=max_terms, tolerance=tolerance)
    return _raw.arrow_matrix_exp(matrix, max_terms=max_terms, tolerance=tolerance)


def arrow_matrix_exp_eigen(matrix):
    if _is_complex_matrix(matrix):
        return arrow_matrix_exp_eigen_complex(matrix)
    return _raw.arrow_matrix_exp_eigen(matrix)


def arrow_matrix_log_taylor(matrix, max_terms=None, tolerance=None):
    _require_real("arrow_matrix_log_taylor", matrix=matrix)
    return _raw.arrow_matrix_log_taylor(matrix, max_terms=max_terms, tolerance=tolerance)


def arrow_matrix_log_eigen(matrix):
    if _is_complex_matrix(matrix):
        return arrow_matrix_log_eigen_complex(matrix)
    return _raw.arrow_matrix_log_eigen(matrix)


def arrow_matrix_log_svd(matrix):
    if _is_complex_matrix(matrix):
        return arrow_matrix_log_svd_complex(matrix)
    return _raw.arrow_matrix_log_svd(matrix)


def arrow_matrix_power(matrix, power):
    if _is_complex_matrix(matrix):
        return arrow_matrix_power_complex(matrix, power)
    return _raw.arrow_matrix_power(matrix, power)


def arrow_matrix_sign(matrix):
    if _is_complex_matrix(matrix):
        return arrow_matrix_sign_complex(matrix)
    return _raw.arrow_matrix_sign(matrix)


def arrow_compute_pca(matrix, n_components=None) -> PcaResult:
    if _is_complex_matrix(matrix):
        return arrow_compute_pca_complex(matrix, n_components=n_components)
    return _pca_result(_raw.arrow_compute_pca(matrix, n_components=n_components))


def arrow_pca_transform(matrix, result: PcaResult):
    if np.iscomplexobj(result.components):
        return arrow_pca_transform_complex(matrix, result)
    return _raw.arrow_pca_transform(matrix, result.components, result.mean)


def arrow_pca_inverse_transform(scores, result: PcaResult):
    if np.iscomplexobj(result.components):
        return arrow_pca_inverse_transform_complex(scores, result)
    return _raw.arrow_pca_inverse_transform(scores, result.components, result.mean)


def arrow_linear_regression(x, y, add_intercept=True) -> RegressionResult:
    if _complex_mode("arrow_linear_regression", x=x, y=y):
        return arrow_linear_regression_complex(x, y, add_intercept=add_intercept)
    return _regression_result(_raw.arrow_linear_regression(x, y, add_intercept=add_intercept))


def arrow_conjugate_gradient(matrix, rhs, tolerance=None, max_iterations=None, *, config: IterativeConfig | None = None):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if _complex_mode("arrow_conjugate_gradient", matrix=matrix, rhs=rhs):
        return arrow_conjugate_gradient_complex(matrix, rhs, **kwargs)
    return _raw.arrow_conjugate_gradient(matrix, rhs, **kwargs)


def arrow_conjugate_gradient_complex(matrix, rhs, tolerance=None, max_iterations=None, *, config: IterativeConfig | None = None):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_complex("arrow_conjugate_gradient_complex", matrix=matrix, rhs=rhs)
    field, storage = _raw.arrow_conjugate_gradient_complex(
        _complex_matrix_storage(matrix),
        _complex_vector_field("rhs"),
        _complex_vector_storage(rhs),
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_gmres(matrix, rhs, tolerance=None, max_iterations=None, *, config: IterativeConfig | None = None):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if _complex_mode("arrow_gmres", matrix=matrix, rhs=rhs):
        return arrow_gmres_complex(matrix, rhs, **kwargs)
    return _raw.arrow_gmres(matrix, rhs, **kwargs)


def arrow_gmres_complex(matrix, rhs, tolerance=None, max_iterations=None, *, config: IterativeConfig | None = None):
    kwargs = _resolve_config(
        config,
        IterativeConfig,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_complex("arrow_gmres_complex", matrix=matrix, rhs=rhs)
    field, storage = _raw.arrow_gmres_complex(
        _complex_matrix_storage(matrix),
        _complex_vector_field("rhs"),
        _complex_vector_storage(rhs),
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_numerical_jacobian(function, x, step_size=None, tolerance=None, max_iterations=None, *, config: JacobianConfig | None = None):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_real("arrow_numerical_jacobian", x=x)
    return _raw.arrow_numerical_jacobian(function, x, **kwargs)


def arrow_numerical_jacobian_central(function, x, step_size=None, tolerance=None, max_iterations=None, *, config: JacobianConfig | None = None):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_real("arrow_numerical_jacobian_central", x=x)
    return _raw.arrow_numerical_jacobian_central(function, x, **kwargs)


def arrow_numerical_gradient(function, x, step_size=None, tolerance=None, max_iterations=None, *, config: JacobianConfig | None = None):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_real("arrow_numerical_gradient", x=x)
    return _raw.arrow_numerical_gradient(function, x, **kwargs)


def arrow_numerical_hessian(function, x, step_size=None, tolerance=None, max_iterations=None, *, config: JacobianConfig | None = None):
    kwargs = _resolve_config(
        config,
        JacobianConfig,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _require_real("arrow_numerical_hessian", x=x)
    return _raw.arrow_numerical_hessian(function, x, **kwargs)


def arrow_backtracking_line_search(
    point,
    direction,
    objective,
    gradient,
    initial_step=None,
    contraction=None,
    sufficient_decrease=None,
    max_iterations=None,
    *,
    config: LineSearchConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        LineSearchConfig,
        initial_step=initial_step,
        contraction=contraction,
        sufficient_decrease=sufficient_decrease,
        max_iterations=max_iterations,
    )
    if _complex_mode("arrow_backtracking_line_search", point=point, direction=direction):
        return arrow_backtracking_line_search_complex(point, direction, objective, gradient, **kwargs)
    return _raw.arrow_backtracking_line_search(point, direction, objective, gradient, **kwargs)


def arrow_backtracking_line_search_complex(
    point,
    direction,
    objective,
    gradient,
    initial_step=None,
    contraction=None,
    sufficient_decrease=None,
    max_iterations=None,
    *,
    config: LineSearchConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        LineSearchConfig,
        initial_step=initial_step,
        contraction=contraction,
        sufficient_decrease=sufficient_decrease,
        max_iterations=max_iterations,
    )
    _require_complex("arrow_backtracking_line_search_complex", point=point, direction=direction)
    return _raw.arrow_backtracking_line_search_complex(
        _complex_vector_field("point"),
        _complex_vector_storage(point),
        _complex_vector_field("direction"),
        _complex_vector_storage(direction),
        objective,
        gradient,
        **kwargs,
    )


def arrow_gradient_descent(initial, objective, gradient, learning_rate=None, max_iterations=None, tolerance=None, *, config: GradientDescentConfig | None = None):
    kwargs = _resolve_config(
        config,
        GradientDescentConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_gradient_descent_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_gradient_descent", initial=initial)
    return _raw.arrow_gradient_descent(initial, objective, gradient, **kwargs)


def arrow_gradient_descent_complex(initial, objective, gradient, learning_rate=None, max_iterations=None, tolerance=None, *, config: GradientDescentConfig | None = None):
    kwargs = _resolve_config(
        config,
        GradientDescentConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_gradient_descent_complex", initial=initial)
    field, storage = _raw.arrow_gradient_descent_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_adam(
    initial,
    objective,
    gradient,
    learning_rate=None,
    beta1=None,
    beta2=None,
    epsilon=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: AdamConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        AdamConfig,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_adam_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_adam", initial=initial)
    return _raw.arrow_adam(initial, objective, gradient, **kwargs)


def arrow_adam_complex(
    initial,
    objective,
    gradient,
    learning_rate=None,
    beta1=None,
    beta2=None,
    epsilon=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: AdamConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        AdamConfig,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_adam_complex", initial=initial)
    field, storage = _raw.arrow_adam_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_momentum_descent(
    initial,
    objective,
    gradient,
    learning_rate=None,
    momentum=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: MomentumConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        MomentumConfig,
        learning_rate=learning_rate,
        momentum=momentum,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_momentum_descent_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_momentum_descent", initial=initial)
    return _raw.arrow_momentum_descent(initial, objective, gradient, **kwargs)


def arrow_momentum_descent_complex(
    initial,
    objective,
    gradient,
    learning_rate=None,
    momentum=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: MomentumConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        MomentumConfig,
        learning_rate=learning_rate,
        momentum=momentum,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_momentum_descent_complex", initial=initial)
    field, storage = _raw.arrow_momentum_descent_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_rmsprop(
    initial,
    objective,
    gradient,
    learning_rate=None,
    rho=None,
    epsilon=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: RMSPropConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        RMSPropConfig,
        learning_rate=learning_rate,
        rho=rho,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_rmsprop_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_rmsprop", initial=initial)
    return _raw.arrow_rmsprop(initial, objective, gradient, **kwargs)


def arrow_rmsprop_complex(
    initial,
    objective,
    gradient,
    learning_rate=None,
    rho=None,
    epsilon=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: RMSPropConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        RMSPropConfig,
        learning_rate=learning_rate,
        rho=rho,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_rmsprop_complex", initial=initial)
    field, storage = _raw.arrow_rmsprop_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_projected_gradient_descent_box(
    initial,
    objective,
    gradient,
    lower_bounds,
    upper_bounds,
    learning_rate=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: ProjectedGradientConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        ProjectedGradientConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _complex_mode(
        "arrow_projected_gradient_descent_box",
        initial=initial,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    ):
        return arrow_projected_gradient_descent_box_complex(
            initial,
            objective,
            gradient,
            lower_bounds,
            upper_bounds,
            **kwargs,
        )
    return _raw.arrow_projected_gradient_descent_box(
        initial,
        objective,
        gradient,
        lower_bounds,
        upper_bounds,
        **kwargs,
    )


def arrow_projected_gradient_descent_box_complex(
    initial,
    objective,
    gradient,
    lower_bounds,
    upper_bounds,
    learning_rate=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: ProjectedGradientConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        ProjectedGradientConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex(
        "arrow_projected_gradient_descent_box_complex",
        initial=initial,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    field, storage = _raw.arrow_projected_gradient_descent_box_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        _complex_vector_storage(lower_bounds),
        _complex_vector_storage(upper_bounds),
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_stochastic_gradient_descent(
    initial,
    stochastic_gradient,
    learning_rate=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: GradientDescentConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        GradientDescentConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_stochastic_gradient_descent_complex(initial, stochastic_gradient, **kwargs)
    _require_real("arrow_stochastic_gradient_descent", initial=initial)
    return _raw.arrow_stochastic_gradient_descent(initial, stochastic_gradient, **kwargs)


def arrow_stochastic_gradient_descent_complex(
    initial,
    stochastic_gradient,
    learning_rate=None,
    max_iterations=None,
    tolerance=None,
    *,
    config: GradientDescentConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        GradientDescentConfig,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    _require_complex("arrow_stochastic_gradient_descent_complex", initial=initial)
    field, storage = _raw.arrow_stochastic_gradient_descent_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        stochastic_gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_bfgs(
    initial,
    objective,
    gradient,
    step_size=None,
    max_iterations=None,
    tolerance=None,
    curvature_tolerance=None,
    *,
    config: BFGSConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        BFGSConfig,
        step_size=step_size,
        max_iterations=max_iterations,
        tolerance=tolerance,
        curvature_tolerance=curvature_tolerance,
    )
    if _is_complex_vector(initial):
        return arrow_bfgs_complex(initial, objective, gradient, **kwargs)
    _require_real("arrow_bfgs", initial=initial)
    return _raw.arrow_bfgs(initial, objective, gradient, **kwargs)


def arrow_bfgs_complex(
    initial,
    objective,
    gradient,
    step_size=None,
    max_iterations=None,
    tolerance=None,
    curvature_tolerance=None,
    *,
    config: BFGSConfig | None = None,
):
    kwargs = _resolve_config(
        config,
        BFGSConfig,
        step_size=step_size,
        max_iterations=max_iterations,
        tolerance=tolerance,
        curvature_tolerance=curvature_tolerance,
    )
    _require_complex("arrow_bfgs_complex", initial=initial)
    field, storage = _raw.arrow_bfgs_complex(
        _complex_vector_field("initial"),
        _complex_vector_storage(initial),
        objective,
        gradient,
        **kwargs,
    )
    return _extension_array(field, storage)


def arrow_sparse_matvec(matrix, rhs):
    _require_csr_matrix_array("arrow_sparse_matvec", matrix)
    return _raw.arrow_sparse_matvec(_arrow_field(matrix, "matrix"), matrix.storage, rhs)


def arrow_sparse_matmat_dense(matrix, dense):
    _require_csr_matrix_array("arrow_sparse_matmat_dense", matrix)
    return _raw.arrow_sparse_matmat_dense(_arrow_field(matrix, "matrix"), matrix.storage, dense)


def arrow_sparse_lu_solve(matrix, rhs):
    _require_csr_matrix_array("arrow_sparse_lu_solve", matrix)
    return _raw.arrow_sparse_lu_solve(_arrow_field(matrix, "matrix"), matrix.storage, rhs)


def arrow_sparse_jacobi_solve(matrix, rhs, tolerance=None, max_iterations=None):
    _require_csr_matrix_array("arrow_sparse_jacobi_solve", matrix)
    return _raw.arrow_sparse_jacobi_solve(
        _arrow_field(matrix, "matrix"),
        matrix.storage,
        rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def arrow_sparse_gauss_seidel_solve(matrix, rhs, tolerance=None, max_iterations=None):
    _require_csr_matrix_array("arrow_sparse_gauss_seidel_solve", matrix)
    return _raw.arrow_sparse_gauss_seidel_solve(
        _arrow_field(matrix, "matrix"),
        matrix.storage,
        rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def arrow_sparse_conjugate_gradient_solve(matrix, rhs, tolerance=None, max_iterations=None):
    _require_csr_matrix_array("arrow_sparse_conjugate_gradient_solve", matrix)
    return _raw.arrow_sparse_conjugate_gradient_solve(
        _arrow_field(matrix, "matrix"),
        matrix.storage,
        rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def arrow_sparse_pcg_solve(matrix, rhs, tolerance=None, max_iterations=None):
    _require_csr_matrix_array("arrow_sparse_pcg_solve", matrix)
    return _raw.arrow_sparse_pcg_solve(
        _arrow_field(matrix, "matrix"),
        matrix.storage,
        rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def arrow_sparse_batched_matvec(matrix, batch_vectors):
    _require_csr_matrix_array("arrow_sparse_batched_matvec", matrix)
    return _raw.arrow_sparse_batched_matvec(
        _arrow_field(matrix, "matrix"),
        matrix.storage,
        batch_vectors,
    )


def arrow_sparse_transpose(matrix) -> CsrMatrix:
    _require_csr_matrix_array("arrow_sparse_transpose", matrix)
    return CsrMatrix.from_components(
        *_raw.arrow_sparse_transpose(_arrow_field(matrix, "matrix"), matrix.storage)
    )


def arrow_sparse_csr_to_csc(matrix) -> CscMatrix:
    _require_csr_matrix_array("arrow_sparse_csr_to_csc", matrix)
    return CscMatrix.from_components(
        *_raw.arrow_sparse_csr_to_csc(_arrow_field(matrix, "matrix"), matrix.storage)
    )


def arrow_sparse_matmat_sparse(left, right) -> CsrMatrix:
    _require_csr_matrix_array("arrow_sparse_matmat_sparse", left)
    _require_csr_matrix_array("arrow_sparse_matmat_sparse", right)
    return CsrMatrix.from_components(
        *_raw.arrow_sparse_matmat_sparse(
            _arrow_field(left, "left"),
            left.storage,
            _arrow_field(right, "right"),
            right.storage,
        )
    )


def arrow_sparse_jacobi_preconditioner(matrix) -> JacobiPreconditioner:
    _require_csr_matrix_array("arrow_sparse_jacobi_preconditioner", matrix)
    return JacobiPreconditioner(
        _raw.arrow_sparse_jacobi_preconditioner(_arrow_field(matrix, "matrix"), matrix.storage)
    )


def arrow_sparse_apply_jacobi_preconditioner(preconditioner, rhs):
    if not isinstance(preconditioner, JacobiPreconditioner):
        raise TypeError("preconditioner must be JacobiPreconditioner")
    return _raw.arrow_sparse_apply_jacobi_preconditioner(preconditioner._raw, rhs)


def arrow_sparse_ilu0_factor(matrix) -> ILU0Factorization:
    _require_csr_matrix_array("arrow_sparse_ilu0_factor", matrix)
    base = arrow_csr_matrix_from_array(matrix)
    raw = _raw.arrow_sparse_ilu0_factor(_arrow_field(matrix, "matrix"), matrix.storage)
    return ILU0Factorization(base, raw)


def arrow_sparse_apply_ilu0_preconditioner(factorization, rhs):
    if not isinstance(factorization, ILU0Factorization):
        raise TypeError("factorization must be ILU0Factorization")
    return _raw.arrow_sparse_apply_ilu0_preconditioner(factorization._raw, rhs)


def arrow_sparse_ilut_factor(
    matrix,
    *,
    drop_tolerance=None,
    max_fill=None,
    config: ILUTConfig | None = None,
) -> ILUTFactorization:
    _require_csr_matrix_array("arrow_sparse_ilut_factor", matrix)
    resolved = _resolve_ilut_config(
        config=config,
        drop_tolerance=drop_tolerance,
        max_fill=max_fill,
    )
    base = arrow_csr_matrix_from_array(matrix)
    raw = _raw.arrow_sparse_ilut_factor(
        _arrow_field(matrix, "matrix"),
        matrix.storage,
        drop_tolerance=resolved.drop_tolerance,
        max_fill=resolved.max_fill,
    )
    return ILUTFactorization(base, raw, config=resolved)


def arrow_sparse_apply_ilut_preconditioner(factorization, rhs):
    if not isinstance(factorization, ILUTFactorization):
        raise TypeError("factorization must be ILUTFactorization")
    return _raw.arrow_sparse_apply_ilut_preconditioner(factorization._raw, rhs)


def arrow_sparse_iluk_factor(
    matrix,
    *,
    level_of_fill=None,
    config: ILUKConfig | None = None,
) -> ILUKFactorization:
    _require_csr_matrix_array("arrow_sparse_iluk_factor", matrix)
    resolved = _resolve_iluk_config(config=config, level_of_fill=level_of_fill)
    base = arrow_csr_matrix_from_array(matrix)
    raw = _raw.arrow_sparse_iluk_factor(
        _arrow_field(matrix, "matrix"),
        matrix.storage,
        level_of_fill=resolved.level_of_fill,
    )
    return ILUKFactorization(base, raw, config=resolved)


def arrow_sparse_apply_iluk_preconditioner(factorization, rhs):
    if not isinstance(factorization, ILUKFactorization):
        raise TypeError("factorization must be ILUKFactorization")
    return _raw.arrow_sparse_apply_iluk_preconditioner(factorization._raw, rhs)


def arrow_sparse_ic0_factor(matrix) -> IC0Factorization:
    _require_csr_matrix_array("arrow_sparse_ic0_factor", matrix)
    base = arrow_csr_matrix_from_array(matrix)
    raw = _raw.arrow_sparse_ic0_factor(_arrow_field(matrix, "matrix"), matrix.storage)
    return IC0Factorization(base, raw)


def arrow_sparse_apply_ic0_preconditioner(factorization, rhs):
    if not isinstance(factorization, IC0Factorization):
        raise TypeError("factorization must be IC0Factorization")
    return _raw.arrow_sparse_apply_ic0_preconditioner(factorization._raw, rhs)


def arrow_sparse_ildl0_factor(matrix) -> ILDL0Factorization:
    _require_csr_matrix_array("arrow_sparse_ildl0_factor", matrix)
    base = arrow_csr_matrix_from_array(matrix)
    raw = _raw.arrow_sparse_ildl0_factor(_arrow_field(matrix, "matrix"), matrix.storage)
    return ILDL0Factorization(base, raw)


def arrow_sparse_apply_ildl0_preconditioner(factorization, rhs):
    if not isinstance(factorization, ILDL0Factorization):
        raise TypeError("factorization must be ILDL0Factorization")
    return _raw.arrow_sparse_apply_ildl0_preconditioner(factorization._raw, rhs)


def arrow_sparse_lu_factor(matrix) -> SparseLUFactorization:
    _require_csr_matrix_array("arrow_sparse_lu_factor", matrix)
    base = arrow_csr_matrix_from_array(matrix)
    raw = _raw.arrow_sparse_lu_factor(_arrow_field(matrix, "matrix"), matrix.storage)
    return SparseLUFactorization(base, raw)


def arrow_sparse_lu_solve_with_factorization(matrix, rhs, factorization):
    _require_csr_matrix_array("arrow_sparse_lu_solve_with_factorization", matrix)
    if not isinstance(factorization, SparseLUFactorization):
        raise TypeError("factorization must be SparseLUFactorization")
    return _raw.arrow_sparse_lu_solve_with_factorization(
        _arrow_field(matrix, "matrix"),
        matrix.storage,
        rhs,
        factorization._raw,
    )


def arrow_sparse_lu_solve_multiple_with_factorization(matrix, rhs, factorization):
    _require_csr_matrix_array("arrow_sparse_lu_solve_multiple_with_factorization", matrix)
    if not isinstance(factorization, SparseLUFactorization):
        raise TypeError("factorization must be SparseLUFactorization")
    return _raw.arrow_sparse_lu_solve_multiple_with_factorization(
        _arrow_field(matrix, "matrix"),
        matrix.storage,
        rhs,
        factorization._raw,
    )


def arrow_sparse_batch_matvec(matrices, vectors):
    _require_csr_matrix_batch_array("arrow_sparse_batch_matvec", matrices)
    _require_variable_shape_tensor_array("arrow_sparse_batch_matvec", vectors, "vectors")
    field, storage = _raw.arrow_sparse_batch_matvec(
        _arrow_field(matrices, "matrices"),
        matrices.storage,
        _arrow_field(vectors, "vectors"),
        vectors.storage,
    )
    return _extension_array(field, storage)


def arrow_sparse_batch_matmat_dense(matrices, right):
    _require_csr_matrix_batch_array("arrow_sparse_batch_matmat_dense", matrices)
    _require_variable_shape_tensor_array("arrow_sparse_batch_matmat_dense", right, "right")
    field, storage = _raw.arrow_sparse_batch_matmat_dense(
        _arrow_field(matrices, "matrices"),
        matrices.storage,
        _arrow_field(right, "right"),
        right.storage,
    )
    return _extension_array(field, storage)


def arrow_sparse_batch_transpose(matrices):
    _require_csr_matrix_batch_array("arrow_sparse_batch_transpose", matrices)
    field, storage = _raw.arrow_sparse_batch_transpose(
        _arrow_field(matrices, "matrices"),
        matrices.storage,
    )
    return _extension_array(field, storage)


def arrow_sparse_batch_matmat_sparse(left, right):
    _require_csr_matrix_batch_array("arrow_sparse_batch_matmat_sparse", left)
    _require_csr_matrix_batch_array("arrow_sparse_batch_matmat_sparse", right)
    field, storage = _raw.arrow_sparse_batch_matmat_sparse(
        _arrow_field(left, "left"),
        left.storage,
        _arrow_field(right, "right"),
        right.storage,
    )
    return _extension_array(field, storage)


__all__ = [
    "arrow_adam",
    "arrow_adam_complex",
    "arrow_backtracking_line_search",
    "arrow_backtracking_line_search_complex",
    "arrow_batched_cholesky",
    "arrow_batched_cosine_distance",
    "arrow_batched_cosine_similarity",
    "arrow_batched_cosine_similarity_complex",
    "arrow_batched_dot",
    "arrow_batched_dot_hermitian",
    "arrow_batched_l2_norm",
    "arrow_batched_l2_norm_complex",
    "arrow_batched_lu",
    "arrow_batched_matmat",
    "arrow_batched_matmat_broadcast_left",
    "arrow_batched_matmat_broadcast_right",
    "arrow_batched_normalize",
    "arrow_batched_normalize_complex",
    "arrow_batched_qr",
    "arrow_batched_row_matvec",
    "arrow_batched_svd",
    "arrow_batched_symmetric_eigen",
    "arrow_bfgs",
    "arrow_bfgs_complex",
    "arrow_center_columns",
    "arrow_center_columns_complex",
    "arrow_cholesky_decompose",
    "arrow_cholesky_decompose_complex",
    "arrow_cholesky_inverse",
    "arrow_cholesky_inverse_complex",
    "arrow_cholesky_solve",
    "arrow_cholesky_solve_complex",
    "arrow_csr_matrix_array",
    "arrow_csr_matrix_batch_array",
    "arrow_csr_matrix_batch_rows",
    "arrow_csr_matrix_from_array",
    "arrow_column_means",
    "arrow_column_means_complex",
    "arrow_conjugate_gradient",
    "arrow_conjugate_gradient_complex",
    "arrow_compute_pca",
    "arrow_compute_pca_complex",
    "arrow_correlation_matrix",
    "arrow_correlation_matrix_complex",
    "arrow_cosine_distance",
    "arrow_cosine_similarity",
    "arrow_cosine_similarity_complex",
    "arrow_covariance_matrix",
    "arrow_covariance_matrix_complex",
    "arrow_dot",
    "arrow_dot_hermitian",
    "arrow_eigen_generalized",
    "arrow_eigen_nonsymmetric",
    "arrow_eigen_nonsymmetric_bi",
    "arrow_eigen_nonsymmetric_complex",
    "arrow_eigen_symmetric",
    "arrow_gram_schmidt",
    "arrow_gram_schmidt_classic",
    "arrow_gram_schmidt_complex",
    "arrow_gmres",
    "arrow_gmres_complex",
    "arrow_gradient_descent",
    "arrow_gradient_descent_complex",
    "arrow_l2_norm",
    "arrow_l2_norm_complex",
    "arrow_linear_regression",
    "arrow_linear_regression_complex",
    "arrow_lu_decompose",
    "arrow_lu_determinant",
    "arrow_lu_determinant_complex",
    "arrow_lu_inverse",
    "arrow_lu_inverse_complex",
    "arrow_lu_log_determinant",
    "arrow_lu_solve",
    "arrow_lu_solve_complex",
    "arrow_matmat",
    "arrow_matmat_complex",
    "arrow_matvec",
    "arrow_matvec_complex",
    "arrow_matrix_exp",
    "arrow_matrix_exp_complex",
    "arrow_matrix_exp_eigen",
    "arrow_matrix_exp_eigen_complex",
    "arrow_matrix_log_eigen",
    "arrow_matrix_log_eigen_complex",
    "arrow_matrix_log_svd",
    "arrow_matrix_log_svd_complex",
    "arrow_matrix_log_taylor",
    "arrow_matrix_power",
    "arrow_matrix_power_complex",
    "arrow_matrix_sign",
    "arrow_matrix_sign_complex",
    "arrow_momentum_descent",
    "arrow_momentum_descent_complex",
    "arrow_numerical_gradient",
    "arrow_numerical_hessian",
    "arrow_numerical_jacobian",
    "arrow_numerical_jacobian_central",
    "arrow_pairwise_cosine_distance",
    "arrow_pairwise_cosine_similarity",
    "arrow_pairwise_l2_distance",
    "arrow_pca_inverse_transform",
    "arrow_pca_inverse_transform_complex",
    "arrow_pca_transform",
    "arrow_pca_transform_complex",
    "arrow_polar_compute",
    "arrow_polar_compute_complex",
    "arrow_projected_gradient_descent_box",
    "arrow_projected_gradient_descent_box_complex",
    "arrow_qr_decompose",
    "arrow_qr_decompose_complex",
    "arrow_qr_decompose_pivoted",
    "arrow_qr_decompose_reduced",
    "arrow_qr_solve_least_squares",
    "arrow_rmsprop",
    "arrow_rmsprop_complex",
    "arrow_schur_compute",
    "arrow_schur_compute_complex",
    "arrow_solve_lower",
    "arrow_solve_lower_complex",
    "arrow_solve_lower_matrix",
    "arrow_sparse_apply_ic0_preconditioner",
    "arrow_sparse_apply_ildl0_preconditioner",
    "arrow_sparse_apply_ilu0_preconditioner",
    "arrow_sparse_apply_iluk_preconditioner",
    "arrow_sparse_apply_ilut_preconditioner",
    "arrow_sparse_apply_jacobi_preconditioner",
    "arrow_sparse_batch_matmat_dense",
    "arrow_sparse_batch_matmat_sparse",
    "arrow_sparse_batch_matvec",
    "arrow_sparse_batch_transpose",
    "arrow_sparse_batched_matvec",
    "arrow_sparse_conjugate_gradient_solve",
    "arrow_sparse_csr_to_csc",
    "arrow_sparse_gauss_seidel_solve",
    "arrow_sparse_ic0_factor",
    "arrow_sparse_ildl0_factor",
    "arrow_sparse_ilu0_factor",
    "arrow_sparse_iluk_factor",
    "arrow_sparse_ilut_factor",
    "arrow_sparse_jacobi_preconditioner",
    "arrow_sparse_jacobi_solve",
    "arrow_sparse_lu_factor",
    "arrow_sparse_lu_solve",
    "arrow_sparse_lu_solve_multiple_with_factorization",
    "arrow_sparse_lu_solve_with_factorization",
    "arrow_sparse_matmat_dense",
    "arrow_sparse_matmat_sparse",
    "arrow_sparse_matvec",
    "arrow_sparse_pcg_solve",
    "arrow_sparse_transpose",
    "arrow_solve_upper",
    "arrow_solve_upper_complex",
    "arrow_solve_upper_matrix",
    "arrow_svd_decompose",
    "arrow_svd_decompose_complex",
    "arrow_svd_decompose_truncated",
    "arrow_svd_decompose_with_tolerance",
    "arrow_svd_null_space",
    "arrow_svd_pseudo_inverse",
    "arrow_stochastic_gradient_descent",
    "arrow_stochastic_gradient_descent_complex",
    "arrow_variable_shape_tensor_array",
    "arrow_variable_shape_tensor_rows",
]
