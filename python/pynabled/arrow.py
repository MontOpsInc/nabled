"""PyArrow bridge for nabled/ndarrow workflows.

Requires ``pyarrow`` plus a `pynabled` build compiled with Arrow support. Published default
wheels include that Rust feature.
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
    CpAls3Result,
    CpAlsNdResult,
    CpAlsReport,
    CpConvergenceReport,
    CpErrorMetrics,
    EigenResult,
    GeneralizedEigenResult,
    HosvdNdResult,
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
    TensorTrainResult,
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
        "Install a default pynabled wheel or rebuild without --no-default-features"
    ) from e


if not hasattr(_raw, "arrow_dot"):
    raise ImportError(
        "pynabled arrow support not available. "
        "Install a default pynabled wheel or rebuild without --no-default-features"
    )


_COMPLEX_EXTENSION_NAMES = {"ndarrow.complex64"}
_COMPLEX_STORAGE_TYPE = pa.list_(pa.field("item", pa.float64(), nullable=False), 2)
_CSR_EXTENSION_NAME = "ndarrow.csr_matrix"
_CSR_BATCH_EXTENSION_NAME = "ndarrow.csr_matrix_batch"
_FIXED_SHAPE_TENSOR_EXTENSION_NAME = "arrow.fixed_shape_tensor"
_VARIABLE_SHAPE_TENSOR_EXTENSION_NAME = "arrow.variable_shape_tensor"
_UINT32_MAX = np.iinfo(np.uint32).max
_INT32_MAX = np.iinfo(np.int32).max


def _list_value_type(type_):
    return type_.value_field.type if hasattr(type_, "value_field") else type_.value_type


class NdarrowComplex64Type(pa.ExtensionType):
    def __init__(self):
        super().__init__(_COMPLEX_STORAGE_TYPE, "ndarrow.complex64")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        del storage_type, serialized
        return cls()

    def __reduce__(self):
        return NdarrowComplex64Type, ()


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


class ArrowFixedShapeTensorType(pa.ExtensionType):
    def __init__(self, value_type, shape):
        self._value_type = value_type
        self._shape = tuple(int(dimension) for dimension in shape)
        if not self._shape:
            raise ValueError("fixed-shape tensor shape must be non-empty")
        element_count = 1
        for dimension in self._shape:
            element_count *= dimension
        value_field = (
            _complex_vector_field("item")
            if value_type == _COMPLEX_STORAGE_TYPE
            else pa.field("item", value_type, nullable=False)
        )
        super().__init__(
            pa.list_(value_field, element_count),
            _FIXED_SHAPE_TENSOR_EXTENSION_NAME,
        )

    def __arrow_ext_serialize__(self):
        return json.dumps(
            {
                "shape": list(self._shape),
                "dim_names": None,
                "permutations": None,
            }
        ).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        metadata = json.loads(serialized.decode()) if serialized else {}
        value_type = storage_type.value_field.type if hasattr(storage_type, "value_field") else storage_type.value_type
        return cls(value_type, metadata["shape"])

    def __reduce__(self):
        return ArrowFixedShapeTensorType, (self._value_type, self._shape)


class ArrowVariableShapeTensorType(pa.ExtensionType):
    def __init__(self, value_type, dimensions: int, uniform_shape):
        self._value_type = value_type
        self._dimensions = int(dimensions)
        self._uniform_shape = None if uniform_shape is None else list(uniform_shape)
        data_value_field = (
            _complex_vector_field("item")
            if value_type == _COMPLEX_STORAGE_TYPE
            else pa.field("item", value_type, nullable=False)
        )
        super().__init__(
            pa.struct(
                [
                    pa.field(
                        "data",
                        pa.list_(data_value_field),
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
_try_register_extension(NdarrowComplex64Type())
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


def _fixed_shape_tensor_type_shape(type_) -> tuple[int, ...]:
    if hasattr(type_, "shape"):
        return tuple(int(dimension) for dimension in type_.shape)
    if hasattr(type_, "_shape"):
        return tuple(int(dimension) for dimension in type_._shape)
    metadata = json.loads(type_.__arrow_ext_serialize__().decode())
    return tuple(int(dimension) for dimension in metadata["shape"])


def _variable_shape_tensor_storage_dimensions(storage_type) -> int:
    shape_type = storage_type.field("shape").type
    return int(shape_type.list_size) if hasattr(shape_type, "list_size") else len(shape_type)


def _variable_shape_tensor_type_uniform_shape(type_):
    if hasattr(type_, "_uniform_shape"):
        return type_._uniform_shape
    marker = "uniform_shape=["
    text = str(type_)
    if marker not in text:
        return None
    start = text.index(marker) + len(marker)
    end = text.index("]", start)
    raw_items = text[start:end].strip()
    if not raw_items:
        return []
    return [
        None if item.strip() in {"null", "None"} else int(item.strip())
        for item in raw_items.split(",")
    ]


def _reduced_variable_shape_tensor_uniform_shape(type_):
    uniform_shape = _variable_shape_tensor_type_uniform_shape(type_)
    if uniform_shape is None:
        return None
    return list(uniform_shape[:-1])


def _variable_shape_tensor_field_from_storage(name, storage, uniform_shape):
    tensor_type = ArrowVariableShapeTensorType(
        _list_value_type(storage.type.field("data").type),
        _variable_shape_tensor_storage_dimensions(storage.type),
        uniform_shape,
    )
    return pa.field(name, tensor_type, nullable=False)


def _arrow_field(array, name):
    if isinstance(array, pa.ExtensionArray) and _extension_name(array.type) in _COMPLEX_EXTENSION_NAMES:
        return _complex_vector_field(name)
    if _is_extension_array_named(array, _CSR_BATCH_EXTENSION_NAME):
        return _csr_matrix_batch_field(name, array.storage.type)
    return pa.field(name, array.type, nullable=False)


def _extension_array(field, storage):
    if _extension_name(field.type) == _FIXED_SHAPE_TENSOR_EXTENSION_NAME:
        try:
            return pa.ExtensionArray.from_storage(field.type, storage)
        except TypeError:
            tensor_type = ArrowFixedShapeTensorType(storage.type.value_field.type, _fixed_shape_tensor_type_shape(field.type))
            return pa.ExtensionArray.from_storage(tensor_type, storage)
    if _extension_name(field.type) == _VARIABLE_SHAPE_TENSOR_EXTENSION_NAME:
        try:
            return pa.ExtensionArray.from_storage(field.type, storage)
        except TypeError:
            tensor_type = ArrowVariableShapeTensorType(
                _list_value_type(storage.type.field("data").type),
                _variable_shape_tensor_storage_dimensions(storage.type),
                _variable_shape_tensor_type_uniform_shape(field.type),
            )
            return pa.ExtensionArray.from_storage(tensor_type, storage)
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


def _complex_storage_array(values):
    np_values = np.asarray(values, dtype=np.complex128)
    primitive = pa.array(np_values.reshape(-1).view(np.float64), type=pa.float64())
    return pa.FixedSizeListArray.from_arrays(primitive, type=_COMPLEX_STORAGE_TYPE)


def _complex_vector_array(values):
    storage = _complex_storage_array(values)
    return pa.ExtensionArray.from_storage(NdarrowComplex64Type(), storage)


def _complex_vector_numpy(array):
    storage = array.storage if isinstance(array, pa.ExtensionArray) else array
    if isinstance(storage, pa.FixedSizeListArray) and storage.type.list_size == 2:
        values = np.asarray(storage.values, dtype=np.float64).reshape(len(storage), 2)
        return values.view(np.complex128).reshape(-1)
    return np.array([complex(real, imag) for real, imag in storage.to_pylist()], dtype=np.complex128)


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


def _require_fixed_shape_tensor_array(name, value, label: str):
    _require_extension_array(name, value, _FIXED_SHAPE_TENSOR_EXTENSION_NAME, label)


def _arrow_real_type_for_dtype(dtype) -> pa.DataType:
    resolved = np.dtype(dtype)
    if resolved == np.dtype(np.float32):
        return pa.float32()
    if resolved == np.dtype(np.float64):
        return pa.float64()
    raise TypeError("dtype must be float32 or float64")


def _arrow_complex_value_type_for_dtype(dtype) -> pa.DataType:
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError("dtype must be complex64 or complex128")
    return _COMPLEX_STORAGE_TYPE


def _numpy_real_dtype_for_arrow(type_) -> np.dtype[np.generic]:
    if pa.types.is_float32(type_):
        return np.dtype(np.float32)
    if pa.types.is_float64(type_):
        return np.dtype(np.float64)
    raise TypeError("expected float32 or float64 Arrow values")


def _numpy_complex_dtype_for_arrow(type_) -> np.dtype[np.generic]:
    if type_ != _COMPLEX_STORAGE_TYPE:
        raise TypeError("expected ndarrow.complex64 Arrow values")
    return np.dtype(np.complex128)


def _is_fixed_shape_tensor_array(value):
    return _is_extension_array_named(value, _FIXED_SHAPE_TENSOR_EXTENSION_NAME)


def _fixed_shape_tensor_storage(value):
    if not _is_fixed_shape_tensor_array(value):
        return None
    return value.storage


def _fixed_shape_tensor_field(name, array):
    _require_fixed_shape_tensor_array("_fixed_shape_tensor_field", array, "array")
    return pa.field(name, array.type, nullable=False)


def _variable_shape_value_type(array):
    if not _is_extension_array_named(array, _VARIABLE_SHAPE_TENSOR_EXTENSION_NAME):
        return None
    return _list_value_type(array.storage.field("data").type)


def _is_complex_fixed_shape_tensor(array):
    storage = _fixed_shape_tensor_storage(array)
    if storage is None:
        return False
    value_type = storage.type.value_field.type
    return value_type == _COMPLEX_STORAGE_TYPE or (
        _is_extension_type(value_type) and _extension_name(value_type) in _COMPLEX_EXTENSION_NAMES
    )


def _is_complex_variable_shape_tensor(array):
    value_type = _variable_shape_value_type(array)
    if value_type is None:
        return False
    return value_type == _COMPLEX_STORAGE_TYPE or (
        _is_extension_type(value_type) and _extension_name(value_type) in _COMPLEX_EXTENSION_NAMES
    )


def _is_complex_tensor(array):
    return _is_complex_fixed_shape_tensor(array) or _is_complex_variable_shape_tensor(array)


def _tensor_mode(name, **arrays):
    flags = {label: _is_complex_tensor(array) for label, array in arrays.items()}
    if any(flags.values()) and not all(flags.values()):
        joined = ", ".join(arrays)
        raise TypeError(
            f"{name} requires {joined} to all be real tensor carriers or all ndarrow.complex64 tensor carriers"
        )
    return any(flags.values())


def _require_real_tensor(name, **arrays):
    wrong = [label for label, array in arrays.items() if _is_complex_tensor(array)]
    if wrong:
        joined = ", ".join(wrong)
        raise TypeError(f"{name} does not currently admit ndarrow.complex64 tensor carriers: {joined}")


def _require_complex_tensor(name, **arrays):
    wrong = [label for label, array in arrays.items() if not _is_complex_tensor(array)]
    if wrong:
        joined = ", ".join(wrong)
        raise TypeError(f"{name} requires ndarrow.complex64 tensor carriers for: {joined}")


def _int32_offsets_from_lengths(lengths, *, label: str):
    resolved = np.asarray(lengths, dtype=np.int64)
    if np.any(resolved < 0):
        raise ValueError(f"{label} lengths must be non-negative")
    offsets = np.empty(len(resolved) + 1, dtype=np.int32)
    offsets[0] = 0
    total = 0
    for index, length in enumerate(resolved, start=1):
        total += int(length)
        if total > _INT32_MAX:
            raise ValueError(f"{label} exceeds int32 offset limits required by Arrow list storage")
        offsets[index] = total
    return offsets


def _shape_array_from_numpy(shape_rows, dimensions: int):
    shapes = np.asarray(shape_rows, dtype=np.int32)
    if shapes.ndim != 2 or shapes.shape[1] != dimensions:
        raise ValueError("shape rows must be a 2D array with the declared tensor rank")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(shapes.reshape(-1), type=pa.int32()),
        type=pa.list_(pa.field("item", pa.int32(), nullable=False), dimensions),
    )


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
    if np.any(csr.indptr > _INT32_MAX):
        raise ValueError("csr row pointers exceed int32 limits required by ndarrow.csr_matrix")
    if np.any(csr.indices > _UINT32_MAX):
        raise ValueError("csr column indices exceed uint32 limits required by ndarrow.csr_matrix")
    value_type = _arrow_real_type_for_dtype(csr.dtype)
    offsets = pa.array(csr.indptr.astype(np.int32, copy=False), type=pa.int32())
    indices = pa.ListArray.from_arrays(
        offsets,
        pa.array(csr.indices.astype(np.uint32, copy=False), type=pa.uint32()),
        type=pa.list_(pa.field("item", pa.uint32(), nullable=False)),
    )
    values = pa.ListArray.from_arrays(
        offsets,
        pa.array(np.asarray(csr.data), type=value_type),
        type=pa.list_(pa.field("item", value_type, nullable=False)),
    )
    storage = pa.StructArray.from_arrays(
        [indices, values],
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
    indices_array = storage.field("indices")
    values_array = storage.field("values")
    dtype = _numpy_real_dtype_for_arrow(_list_value_type(storage.field("values").type))
    ncols = int(matrix.type._ncols if hasattr(matrix.type, "_ncols") else json.loads(matrix.type.__arrow_ext_serialize__().decode())["ncols"])
    indptr = np.asarray(indices_array.offsets, dtype=np.int32)
    if not np.array_equal(indptr, np.asarray(values_array.offsets, dtype=np.int32)):
        raise ValueError("ndarrow.csr_matrix storage has mismatched row lengths")
    flat_indices = np.asarray(indices_array.values, dtype=np.uint32)
    if np.any(flat_indices > _INT32_MAX):
        raise ValueError("ndarrow.csr_matrix indices exceed int32 limits required by pynabled")
    flat_values = np.asarray(values_array.values, dtype=dtype)
    return CsrMatrix.from_components(
        (len(indices_array), ncols),
        indptr,
        flat_indices.astype(np.int32, copy=False),
        flat_values,
    )


def arrow_csr_matrix_batch_array(matrices, *, copy: bool = False, dtype=None, index_dtype=None):
    rows = [CsrMatrix.from_scipy(matrix, copy=copy, dtype=dtype, index_dtype=index_dtype) for matrix in matrices]
    if not rows:
        raise ValueError("arrow_csr_matrix_batch_array requires at least one matrix")
    resolved_dtype = rows[0].dtype
    if any(row.dtype != resolved_dtype for row in rows[1:]):
        raise TypeError("all matrices in a csr_matrix_batch must share dtype float32 or float64")
    shapes = []
    row_ptr_lengths = np.empty(len(rows), dtype=np.int64)
    col_index_lengths = np.empty(len(rows), dtype=np.int64)
    value_lengths = np.empty(len(rows), dtype=np.int64)
    for row in rows:
        if np.any(row.indptr > _INT32_MAX):
            raise ValueError("csr row pointers exceed int32 limits required by ndarrow.csr_matrix_batch")
        if np.any(row.indices > _UINT32_MAX):
            raise ValueError("csr column indices exceed uint32 limits required by ndarrow.csr_matrix_batch")
        shapes.append([row.nrows, row.ncols])
    for index, row in enumerate(rows):
        row_ptr_lengths[index] = row.indptr.size
        col_index_lengths[index] = row.indices.size
        value_lengths[index] = row.data.size
    value_type = _arrow_real_type_for_dtype(resolved_dtype)
    shape_array = _shape_array_from_numpy(np.asarray(shapes, dtype=np.int32), 2)
    row_ptrs = pa.ListArray.from_arrays(
        pa.array(_int32_offsets_from_lengths(row_ptr_lengths, label="csr_matrix_batch row_ptrs"), type=pa.int32()),
        pa.array(
            np.concatenate([row.indptr.astype(np.int32, copy=False) for row in rows]),
            type=pa.int32(),
        ),
        type=pa.list_(pa.field("item", pa.int32(), nullable=False)),
    )
    col_indices = pa.ListArray.from_arrays(
        pa.array(_int32_offsets_from_lengths(col_index_lengths, label="csr_matrix_batch col_indices"), type=pa.int32()),
        pa.array(
            np.concatenate([row.indices.astype(np.uint32, copy=False) for row in rows]),
            type=pa.uint32(),
        ),
        type=pa.list_(pa.field("item", pa.uint32(), nullable=False)),
    )
    values = pa.ListArray.from_arrays(
        pa.array(_int32_offsets_from_lengths(value_lengths, label="csr_matrix_batch values"), type=pa.int32()),
        pa.array(np.concatenate([np.asarray(row.data) for row in rows]), type=value_type),
        type=pa.list_(pa.field("item", value_type, nullable=False)),
    )
    storage = pa.StructArray.from_arrays(
        [shape_array, row_ptrs, col_indices, values],
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
    shape_array = storage.field("shape")
    row_ptrs = storage.field("row_ptrs")
    col_indices = storage.field("col_indices")
    values = storage.field("values")
    dtype = _numpy_real_dtype_for_arrow(_list_value_type(values.type))
    shapes = np.asarray(shape_array.values, dtype=np.int32).reshape(len(shape_array), 2)
    row_ptr_offsets = np.asarray(row_ptrs.offsets, dtype=np.int32)
    row_ptr_values = np.asarray(row_ptrs.values, dtype=np.int32)
    col_offsets = np.asarray(col_indices.offsets, dtype=np.int32)
    col_values = np.asarray(col_indices.values, dtype=np.uint32)
    value_offsets = np.asarray(values.offsets, dtype=np.int32)
    flat_values = np.asarray(values.values, dtype=dtype)
    rows = []
    for index, shape in enumerate(shapes):
        indptr = row_ptr_values[row_ptr_offsets[index] : row_ptr_offsets[index + 1]]
        indices = col_values[col_offsets[index] : col_offsets[index + 1]]
        data = flat_values[value_offsets[index] : value_offsets[index + 1]]
        if indptr.size == 0:
            raise ValueError("ndarrow.csr_matrix_batch storage has empty row_ptrs entry")
        if indptr[-1] != indices.size or indptr[-1] != data.size:
            raise ValueError("ndarrow.csr_matrix_batch storage has mismatched row lengths")
        if np.any(indices > _INT32_MAX):
            raise ValueError("ndarrow.csr_matrix_batch indices exceed int32 limits required by pynabled")
        rows.append(
            CsrMatrix.from_components(
                tuple(int(dimension) for dimension in shape),
                indptr,
                indices.astype(np.int32, copy=False),
                data,
            )
        )
    return rows


def _fixed_shape_tensor_shape(value) -> tuple[int, ...]:
    return _fixed_shape_tensor_type_shape(value.type)


def arrow_fixed_shape_tensor_array(value, *, dtype=None):
    tensor = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    if tensor.ndim < 2:
        raise ValueError("arrow_fixed_shape_tensor_array requires an ndarray with batch axis and tensor rank >= 1")
    element_count = int(np.prod(tensor.shape[1:], dtype=np.int64))
    if np.issubdtype(tensor.dtype, np.floating):
        resolved_dtype = tensor.dtype
        if resolved_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("arrow_fixed_shape_tensor_array currently supports float32 or float64")
        value_type = _arrow_real_type_for_dtype(resolved_dtype)
        storage = pa.FixedSizeListArray.from_arrays(
            pa.array(tensor.reshape(-1), type=value_type),
            type=pa.list_(pa.field("item", value_type, nullable=False), element_count),
        )
        return pa.ExtensionArray.from_storage(
            ArrowFixedShapeTensorType(value_type, tensor.shape[1:]),
            storage,
        )
    if np.issubdtype(tensor.dtype, np.complexfloating):
        tensor = np.asarray(tensor, dtype=np.complex128)
        flat = tensor.reshape(-1)
        storage = pa.FixedSizeListArray.from_arrays(
            _complex_storage_array(flat),
            type=pa.list_(_complex_vector_field("item"), element_count),
        )
        return pa.ExtensionArray.from_storage(
            ArrowFixedShapeTensorType(_COMPLEX_STORAGE_TYPE, tensor.shape[1:]),
            storage,
        )
    raise TypeError(
        "arrow_fixed_shape_tensor_array currently supports float32, float64, complex64, or complex128"
    )


def arrow_fixed_shape_tensor_numpy(value) -> np.ndarray:
    _require_fixed_shape_tensor_array("arrow_fixed_shape_tensor_numpy", value, "value")
    storage = value.storage
    if _is_complex_fixed_shape_tensor(value):
        flat = _complex_vector_numpy(storage.values)
        return flat.reshape((len(storage), *_fixed_shape_tensor_shape(value)))
    flat = np.asarray(storage.values)
    return flat.reshape((len(storage), *_fixed_shape_tensor_shape(value)))


def arrow_variable_shape_tensor_array(rows, *, uniform_shape=None, dtype=None):
    numpy_rows = [np.asarray(row, dtype=dtype) if dtype is not None else np.asarray(row) for row in rows]
    if not numpy_rows:
        raise ValueError("arrow_variable_shape_tensor_array requires at least one tensor")
    rank = numpy_rows[0].ndim
    if rank == 0:
        raise ValueError("arrow_variable_shape_tensor_array requires tensors with rank >= 1")
    resolved_dtype = numpy_rows[0].dtype
    if resolved_dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    ):
        raise TypeError(
            "arrow_variable_shape_tensor_array currently supports float32, float64, complex64, or complex128"
        )
    if any(row.ndim != rank for row in numpy_rows[1:]):
        raise ValueError("all tensors in a variable_shape_tensor batch must share rank")
    if any(row.dtype != resolved_dtype for row in numpy_rows[1:]):
        raise TypeError("all tensors in a variable_shape_tensor batch must share dtype")
    normalized_uniform = None if uniform_shape is None else [None if item is None else int(item) for item in uniform_shape]
    if normalized_uniform is not None and len(normalized_uniform) != rank:
        raise ValueError("uniform_shape length must match tensor rank")
    packed_shapes = np.asarray([list(map(int, row.shape)) for row in numpy_rows], dtype=np.int32)
    packed_lengths = np.asarray([row.size for row in numpy_rows], dtype=np.int64)
    offsets = pa.array(_int32_offsets_from_lengths(packed_lengths, label="variable_shape_tensor data"), type=pa.int32())
    if np.issubdtype(resolved_dtype, np.floating):
        value_type = _arrow_real_type_for_dtype(resolved_dtype)
        flat = np.concatenate([row.reshape(-1) for row in numpy_rows])
        values = pa.array(flat, type=value_type)
        data = pa.ListArray.from_arrays(
            offsets,
            values,
            type=pa.list_(pa.field("item", value_type, nullable=False)),
        )
    else:
        value_type = _arrow_complex_value_type_for_dtype(resolved_dtype)
        flat = np.concatenate([row.reshape(-1).astype(np.complex128, copy=False) for row in numpy_rows])
        values = _complex_storage_array(flat)
        data = pa.ListArray.from_arrays(
            offsets,
            values,
            type=pa.list_(_complex_vector_field("item")),
        )
    storage = pa.StructArray.from_arrays(
        [
            data,
            _shape_array_from_numpy(packed_shapes, rank),
        ],
        fields=[
            pa.field(
                "data",
                data.type,
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
    data = storage.field("data")
    shapes = storage.field("shape")
    rank = _variable_shape_tensor_storage_dimensions(storage.type)
    value_type = _list_value_type(data.type)
    if value_type == _COMPLEX_STORAGE_TYPE or (
        _is_extension_type(value_type) and _extension_name(value_type) in _COMPLEX_EXTENSION_NAMES
    ):
        dtype = np.dtype(np.complex128)
        flat = _complex_vector_numpy(data.values)
    else:
        dtype = _numpy_real_dtype_for_arrow(value_type)
        flat = np.asarray(data.values, dtype=dtype)
    offsets = np.asarray(data.offsets, dtype=np.int32)
    shape_rows = np.asarray(shapes.values, dtype=np.int32).reshape(len(shapes), rank)
    rows = []
    for index, shape in enumerate(shape_rows):
        row = flat[offsets[index] : offsets[index + 1]]
        rows.append(row.reshape(tuple(int(dimension) for dimension in shape)))
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
    if len(raw_result) == 2:
        l, u = raw_result
        return LuResult(l=l, u=u)
    l, u, pivots, permutation_sign = raw_result
    return LuResult(l=l, u=u, pivots=pivots, permutation_sign=permutation_sign)


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


def _cp_metrics(raw_metrics) -> CpErrorMetrics:
    signal_norm, residual_norm, relative_error, fit = raw_metrics
    return CpErrorMetrics(
        signal_norm=float(signal_norm),
        residual_norm=float(residual_norm),
        relative_error=float(relative_error),
        fit=float(fit),
    )


def _cp_report(raw_report) -> CpAlsReport:
    (iterations_run, converged, final_max_factor_change), metrics = raw_report
    return CpAlsReport(
        convergence=CpConvergenceReport(
            iterations_run=int(iterations_run),
            converged=bool(converged),
            final_max_factor_change=float(final_max_factor_change),
        ),
        metrics=_cp_metrics(metrics),
    )


def _cp_als_nd_result(weights, factors) -> CpAlsNdResult:
    shape = tuple(int(factor.shape[0]) for factor in factors)
    return CpAlsNdResult(
        weights=weights,
        factors=list(factors),
        shape=shape,
    )


def _tt_result(cores) -> TensorTrainResult:
    return TensorTrainResult(cores=list(cores))


def arrow_tensor_sum_last_axis(tensor):
    if _is_fixed_shape_tensor_array(tensor):
        field = _fixed_shape_tensor_field("tensor", tensor)
        if _is_complex_fixed_shape_tensor(tensor):
            out_field, out_storage = _raw.arrow_tensor_sum_last_axis_fixed_complex(field, tensor.storage)
        else:
            out_field, out_storage = _raw.arrow_tensor_sum_last_axis_fixed(field, tensor.storage)
        return _extension_array(out_field, out_storage)
    _require_variable_shape_tensor_array("arrow_tensor_sum_last_axis", tensor, "tensor")
    if _is_complex_variable_shape_tensor(tensor):
        out_storage = _raw.arrow_tensor_sum_last_axis_variable_complex_storage(
            _arrow_field(tensor, "tensor"),
            tensor.storage,
        )
        out_field = _variable_shape_tensor_field_from_storage(
            "tensor",
            out_storage,
            _reduced_variable_shape_tensor_uniform_shape(tensor.type),
        )
    else:
        out_field, out_storage = _raw.arrow_tensor_sum_last_axis_variable(
            _arrow_field(tensor, "tensor"),
            tensor.storage,
        )
    return _extension_array(out_field, out_storage)


def arrow_tensor_l2_norm_last_axis(tensor):
    if _is_fixed_shape_tensor_array(tensor):
        field = _fixed_shape_tensor_field("tensor", tensor)
        if _is_complex_fixed_shape_tensor(tensor):
            out_field, out_storage = _raw.arrow_tensor_l2_norm_last_axis_fixed_complex(
                field,
                tensor.storage,
            )
        else:
            out_field, out_storage = _raw.arrow_tensor_l2_norm_last_axis_fixed(field, tensor.storage)
        return _extension_array(out_field, out_storage)
    _require_variable_shape_tensor_array("arrow_tensor_l2_norm_last_axis", tensor, "tensor")
    if _is_complex_variable_shape_tensor(tensor):
        out_storage = _raw.arrow_tensor_l2_norm_last_axis_variable_complex_storage(
            _arrow_field(tensor, "tensor"),
            tensor.storage,
        )
        out_field = _variable_shape_tensor_field_from_storage(
            "tensor",
            out_storage,
            _reduced_variable_shape_tensor_uniform_shape(tensor.type),
        )
    else:
        out_field, out_storage = _raw.arrow_tensor_l2_norm_last_axis_variable(
            _arrow_field(tensor, "tensor"),
            tensor.storage,
        )
    return _extension_array(out_field, out_storage)


def arrow_tensor_normalize_last_axis(tensor):
    if _is_fixed_shape_tensor_array(tensor):
        field = _fixed_shape_tensor_field("tensor", tensor)
        if _is_complex_fixed_shape_tensor(tensor):
            out_field, out_storage = _raw.arrow_tensor_normalize_last_axis_fixed_complex(
                field,
                tensor.storage,
            )
        else:
            out_field, out_storage = _raw.arrow_tensor_normalize_last_axis_fixed(field, tensor.storage)
        return _extension_array(out_field, out_storage)
    _require_variable_shape_tensor_array("arrow_tensor_normalize_last_axis", tensor, "tensor")
    if _is_complex_variable_shape_tensor(tensor):
        out_storage = _raw.arrow_tensor_normalize_last_axis_variable_complex_storage(
            _arrow_field(tensor, "tensor"),
            tensor.storage,
        )
        out_field = _variable_shape_tensor_field_from_storage(
            "tensor",
            out_storage,
            _variable_shape_tensor_type_uniform_shape(tensor.type),
        )
    else:
        out_field, out_storage = _raw.arrow_tensor_normalize_last_axis_variable(
            _arrow_field(tensor, "tensor"),
            tensor.storage,
        )
    return _extension_array(out_field, out_storage)


def arrow_tensor_batched_dot_last_axis(left, right):
    if _is_fixed_shape_tensor_array(left) and _is_fixed_shape_tensor_array(right):
        if _tensor_mode("arrow_tensor_batched_dot_last_axis", left=left, right=right):
            out_field, out_storage = _raw.arrow_tensor_batched_dot_last_axis_fixed_complex(
                _fixed_shape_tensor_field("left", left),
                left.storage,
                _fixed_shape_tensor_field("right", right),
                right.storage,
            )
        else:
            out_field, out_storage = _raw.arrow_tensor_batched_dot_last_axis_fixed(
                _fixed_shape_tensor_field("left", left),
                left.storage,
                _fixed_shape_tensor_field("right", right),
                right.storage,
            )
        return _extension_array(out_field, out_storage)
    _require_variable_shape_tensor_array("arrow_tensor_batched_dot_last_axis", left, "left")
    _require_variable_shape_tensor_array("arrow_tensor_batched_dot_last_axis", right, "right")
    if _tensor_mode("arrow_tensor_batched_dot_last_axis", left=left, right=right):
        out_storage = _raw.arrow_tensor_batched_dot_last_axis_variable_complex_storage(
            _arrow_field(left, "left"),
            left.storage,
            _arrow_field(right, "right"),
            right.storage,
        )
        out_field = _variable_shape_tensor_field_from_storage(
            "left",
            out_storage,
            _reduced_variable_shape_tensor_uniform_shape(left.type),
        )
    else:
        out_field, out_storage = _raw.arrow_tensor_batched_dot_last_axis_variable(
            _arrow_field(left, "left"),
            left.storage,
            _arrow_field(right, "right"),
            right.storage,
        )
    return _extension_array(out_field, out_storage)


def arrow_tensor_permute_axes(tensor, permutation):
    _require_fixed_shape_tensor_array("arrow_tensor_permute_axes", tensor, "tensor")
    if _is_complex_fixed_shape_tensor(tensor):
        out_field, out_storage = _raw.arrow_tensor_permute_axes_complex(
            _fixed_shape_tensor_field("tensor", tensor),
            tensor.storage,
            permutation,
        )
    else:
        out_field, out_storage = _raw.arrow_tensor_permute_axes(
            _fixed_shape_tensor_field("tensor", tensor),
            tensor.storage,
            permutation,
        )
    return _extension_array(out_field, out_storage)


def arrow_tensor_contract_axes(left, right, left_axes, right_axes):
    _require_fixed_shape_tensor_array("arrow_tensor_contract_axes", left, "left")
    _require_fixed_shape_tensor_array("arrow_tensor_contract_axes", right, "right")
    if _tensor_mode("arrow_tensor_contract_axes", left=left, right=right):
        out_field, out_storage = _raw.arrow_tensor_contract_axes_complex(
            _fixed_shape_tensor_field("left", left),
            left.storage,
            _fixed_shape_tensor_field("right", right),
            right.storage,
            left_axes,
            right_axes,
        )
    else:
        out_field, out_storage = _raw.arrow_tensor_contract_axes(
            _fixed_shape_tensor_field("left", left),
            left.storage,
            _fixed_shape_tensor_field("right", right),
            right.storage,
            left_axes,
            right_axes,
        )
    return _extension_array(out_field, out_storage)


def arrow_tensor_batched_matmul_last_two(left, right):
    _require_fixed_shape_tensor_array("arrow_tensor_batched_matmul_last_two", left, "left")
    _require_fixed_shape_tensor_array("arrow_tensor_batched_matmul_last_two", right, "right")
    if _tensor_mode("arrow_tensor_batched_matmul_last_two", left=left, right=right):
        out_field, out_storage = _raw.arrow_tensor_batched_matmul_last_two_complex(
            _fixed_shape_tensor_field("left", left),
            left.storage,
            _fixed_shape_tensor_field("right", right),
            right.storage,
        )
    else:
        out_field, out_storage = _raw.arrow_tensor_batched_matmul_last_two(
            _fixed_shape_tensor_field("left", left),
            left.storage,
            _fixed_shape_tensor_field("right", right),
            right.storage,
        )
    return _extension_array(out_field, out_storage)


def arrow_tensor_cube_matvec(cube, vectors):
    _require_fixed_shape_tensor_array("arrow_tensor_cube_matvec", cube, "cube")
    if _is_complex_fixed_shape_tensor(cube):
        _require_complex("arrow_tensor_cube_matvec", vectors=vectors)
        return _raw.arrow_tensor_cube_matvec_complex(
            _fixed_shape_tensor_field("cube", cube),
            cube.storage,
            vectors,
        )
    _require_real_tensor("arrow_tensor_cube_matvec", cube=cube)
    _require_real("arrow_tensor_cube_matvec", vectors=vectors)
    return _raw.arrow_tensor_cube_matvec(
        _fixed_shape_tensor_field("cube", cube),
        cube.storage,
        vectors,
    )


def arrow_tensor_cube_matmat(left, right):
    _require_fixed_shape_tensor_array("arrow_tensor_cube_matmat", left, "left")
    _require_fixed_shape_tensor_array("arrow_tensor_cube_matmat", right, "right")
    if _tensor_mode("arrow_tensor_cube_matmat", left=left, right=right):
        out_field, out_storage = _raw.arrow_tensor_cube_matmat_complex(
            _fixed_shape_tensor_field("left", left),
            left.storage,
            _fixed_shape_tensor_field("right", right),
            right.storage,
        )
    else:
        out_field, out_storage = _raw.arrow_tensor_cube_matmat(
            _fixed_shape_tensor_field("left", left),
            left.storage,
            _fixed_shape_tensor_field("right", right),
            right.storage,
        )
    return _extension_array(out_field, out_storage)


def arrow_tensor_flatten_cubes(tensor):
    _require_fixed_shape_tensor_array("arrow_tensor_flatten_cubes", tensor, "tensor")
    _require_real_tensor("arrow_tensor_flatten_cubes", tensor=tensor)
    return _raw.arrow_tensor_flatten_cubes(_fixed_shape_tensor_field("tensor", tensor), tensor.storage)


def arrow_tensor_einsum(expression, left, right):
    _require_fixed_shape_tensor_array("arrow_tensor_einsum", left, "left")
    _require_fixed_shape_tensor_array("arrow_tensor_einsum", right, "right")
    if _tensor_mode("arrow_tensor_einsum", left=left, right=right):
        out_field, out_storage = _raw.arrow_tensor_einsum_complex(
            expression,
            _fixed_shape_tensor_field("left", left),
            left.storage,
            _fixed_shape_tensor_field("right", right),
            right.storage,
        )
    else:
        out_field, out_storage = _raw.arrow_tensor_einsum(
            expression,
            _fixed_shape_tensor_field("left", left),
            left.storage,
            _fixed_shape_tensor_field("right", right),
            right.storage,
        )
    return _extension_array(out_field, out_storage)


def arrow_tensor_cp_als3(tensor, rank, max_iterations=None, tolerance=None) -> CpAls3Result:
    _require_fixed_shape_tensor_array("arrow_tensor_cp_als3", tensor, "tensor")
    _require_real_tensor("arrow_tensor_cp_als3", tensor=tensor)
    weights, factor_0, factor_1, factor_2 = _raw.arrow_tensor_cp_als3(
        _fixed_shape_tensor_field("tensor", tensor),
        tensor.storage,
        rank,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return CpAls3Result(
        weights=weights,
        factor_0=factor_0,
        factor_1=factor_1,
        factor_2=factor_2,
    )


def arrow_tensor_cp_als3_with_report(tensor, rank, max_iterations=None, tolerance=None):
    _require_fixed_shape_tensor_array("arrow_tensor_cp_als3_with_report", tensor, "tensor")
    _require_real_tensor("arrow_tensor_cp_als3_with_report", tensor=tensor)
    raw_result, raw_report = _raw.arrow_tensor_cp_als3_with_report(
        _fixed_shape_tensor_field("tensor", tensor),
        tensor.storage,
        rank,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    weights, factor_0, factor_1, factor_2 = raw_result
    return (
        CpAls3Result(
            weights=weights,
            factor_0=factor_0,
            factor_1=factor_1,
            factor_2=factor_2,
        ),
        _cp_report(raw_report),
    )


def arrow_tensor_cp_als3_diagnostics(tensor, result: CpAls3Result) -> CpErrorMetrics:
    _require_fixed_shape_tensor_array("arrow_tensor_cp_als3_diagnostics", tensor, "tensor")
    _require_real_tensor("arrow_tensor_cp_als3_diagnostics", tensor=tensor)
    return _cp_metrics(
        _raw.arrow_tensor_cp_als3_diagnostics(
            _fixed_shape_tensor_field("tensor", tensor),
            tensor.storage,
            result.weights,
            [result.factor_0, result.factor_1, result.factor_2],
        )
    )


def arrow_tensor_cp_als3_reconstruct(result: CpAls3Result, *, field_name: str = "tensor"):
    out_field, out_storage = _raw.arrow_tensor_cp_als3_reconstruct(
        field_name,
        result.weights,
        [result.factor_0, result.factor_1, result.factor_2],
    )
    return _extension_array(out_field, out_storage)


def arrow_tensor_cp_als_nd(tensor, rank, max_iterations=None, tolerance=None) -> CpAlsNdResult:
    _require_fixed_shape_tensor_array("arrow_tensor_cp_als_nd", tensor, "tensor")
    _require_real_tensor("arrow_tensor_cp_als_nd", tensor=tensor)
    weights, factors = _raw.arrow_tensor_cp_als_nd(
        _fixed_shape_tensor_field("tensor", tensor),
        tensor.storage,
        rank,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return _cp_als_nd_result(weights, factors)


def arrow_tensor_cp_als_nd_with_report(tensor, rank, max_iterations=None, tolerance=None):
    _require_fixed_shape_tensor_array("arrow_tensor_cp_als_nd_with_report", tensor, "tensor")
    _require_real_tensor("arrow_tensor_cp_als_nd_with_report", tensor=tensor)
    raw_result, raw_report = _raw.arrow_tensor_cp_als_nd_with_report(
        _fixed_shape_tensor_field("tensor", tensor),
        tensor.storage,
        rank,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    weights, factors = raw_result
    return (_cp_als_nd_result(weights, factors), _cp_report(raw_report))


def arrow_tensor_cp_als_nd_diagnostics(tensor, result: CpAlsNdResult) -> CpErrorMetrics:
    _require_fixed_shape_tensor_array("arrow_tensor_cp_als_nd_diagnostics", tensor, "tensor")
    _require_real_tensor("arrow_tensor_cp_als_nd_diagnostics", tensor=tensor)
    return _cp_metrics(
        _raw.arrow_tensor_cp_als_nd_diagnostics(
            _fixed_shape_tensor_field("tensor", tensor),
            tensor.storage,
            result.weights,
            result.factors,
        )
    )


def arrow_tensor_cp_als_nd_reconstruct(result: CpAlsNdResult, *, field_name: str = "tensor"):
    out_field, out_storage = _raw.arrow_tensor_cp_als_nd_reconstruct(
        field_name,
        result.weights,
        result.factors,
    )
    return _extension_array(out_field, out_storage)


def arrow_tensor_hosvd_nd(tensor, ranks) -> HosvdNdResult:
    _require_fixed_shape_tensor_array("arrow_tensor_hosvd_nd", tensor, "tensor")
    _require_real_tensor("arrow_tensor_hosvd_nd", tensor=tensor)
    core, factors = _raw.arrow_tensor_hosvd_nd(
        _fixed_shape_tensor_field("tensor", tensor),
        tensor.storage,
        ranks,
    )
    return HosvdNdResult(core=core, factors=list(factors))


def arrow_tensor_hooi_nd(tensor, ranks, max_iterations=None, tolerance=None) -> HosvdNdResult:
    _require_fixed_shape_tensor_array("arrow_tensor_hooi_nd", tensor, "tensor")
    _require_real_tensor("arrow_tensor_hooi_nd", tensor=tensor)
    core, factors = _raw.arrow_tensor_hooi_nd(
        _fixed_shape_tensor_field("tensor", tensor),
        tensor.storage,
        ranks,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return HosvdNdResult(core=core, factors=list(factors))


def arrow_tensor_hosvd_nd_reconstruct(result: HosvdNdResult, *, field_name: str = "tensor"):
    out_field, out_storage = _raw.arrow_tensor_hosvd_nd_reconstruct(
        field_name,
        result.core,
        result.factors,
    )
    return _extension_array(out_field, out_storage)


def arrow_tensor_tucker_project(tensor, result: HosvdNdResult):
    _require_fixed_shape_tensor_array("arrow_tensor_tucker_project", tensor, "tensor")
    _require_real_tensor("arrow_tensor_tucker_project", tensor=tensor)
    out_field, out_storage = _raw.arrow_tensor_tucker_project(
        _fixed_shape_tensor_field("tensor", tensor),
        tensor.storage,
        result.factors,
    )
    return _extension_array(out_field, out_storage)


def arrow_tensor_tucker_expand(result: HosvdNdResult, *, field_name: str = "tensor"):
    core = arrow_fixed_shape_tensor_array(np.asarray(result.core))
    out_field, out_storage = _raw.arrow_tensor_tucker_expand(
        _fixed_shape_tensor_field(field_name, core),
        core.storage,
        result.factors,
    )
    return _extension_array(out_field, out_storage)


def arrow_tensor_tt_svd(tensor, max_rank=None, tolerance=None) -> TensorTrainResult:
    _require_fixed_shape_tensor_array("arrow_tensor_tt_svd", tensor, "tensor")
    _require_real_tensor("arrow_tensor_tt_svd", tensor=tensor)
    return _tt_result(
        _raw.arrow_tensor_tt_svd(
            _fixed_shape_tensor_field("tensor", tensor),
            tensor.storage,
            max_rank=max_rank,
            tolerance=tolerance,
        )
    )


def arrow_tensor_tt_orthogonalize_left(result: TensorTrainResult) -> TensorTrainResult:
    return _tt_result(_raw.arrow_tensor_tt_orthogonalize_left(result.cores))


def arrow_tensor_tt_orthogonalize_right(result: TensorTrainResult) -> TensorTrainResult:
    return _tt_result(_raw.arrow_tensor_tt_orthogonalize_right(result.cores))


def arrow_tensor_tt_round(result: TensorTrainResult, max_rank=None, tolerance=None) -> TensorTrainResult:
    return _tt_result(
        _raw.arrow_tensor_tt_round(result.cores, max_rank=max_rank, tolerance=tolerance)
    )


def arrow_tensor_tt_inner(left: TensorTrainResult, right: TensorTrainResult):
    return _raw.arrow_tensor_tt_inner(left.cores, right.cores)


def arrow_tensor_tt_norm(result: TensorTrainResult):
    return _raw.arrow_tensor_tt_norm(result.cores)


def arrow_tensor_tt_add(left: TensorTrainResult, right: TensorTrainResult) -> TensorTrainResult:
    return _tt_result(_raw.arrow_tensor_tt_add(left.cores, right.cores))


def arrow_tensor_tt_hadamard(left: TensorTrainResult, right: TensorTrainResult) -> TensorTrainResult:
    return _tt_result(_raw.arrow_tensor_tt_hadamard(left.cores, right.cores))


def arrow_tensor_tt_hadamard_round(
    left: TensorTrainResult,
    right: TensorTrainResult,
    max_rank=None,
    tolerance=None,
) -> TensorTrainResult:
    return _tt_result(
        _raw.arrow_tensor_tt_hadamard_round(
            left.cores,
            right.cores,
            max_rank=max_rank,
            tolerance=tolerance,
        )
    )


def arrow_tensor_tt_svd_reconstruct(result: TensorTrainResult, *, field_name: str = "tensor"):
    out_field, out_storage = _raw.arrow_tensor_tt_svd_reconstruct(field_name, result.cores)
    return _extension_array(out_field, out_storage)


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


_HAS_ARROW_EMBEDDINGS = hasattr(_raw, "arrow_embeddings_query_corpus_scores")


def _require_arrow_embeddings(name):
    if not _HAS_ARROW_EMBEDDINGS:
        raise ImportError(
            f"{name} requires a pynabled build compiled with the 'embeddings' feature"
        )


def arrow_embeddings_query_corpus_scores(queries, corpus, metric="cosine"):
    """Score every query row against every corpus row, returning a FixedSizeList matrix."""
    _require_arrow_embeddings("arrow_embeddings_query_corpus_scores")
    _require_real("arrow_embeddings_query_corpus_scores", queries=queries, corpus=corpus)
    return _raw.arrow_embeddings_query_corpus_scores(queries, corpus, metric)


def arrow_embeddings_rerank(query, candidates, k, metric="cosine"):
    """Rerank ``candidates`` against a single ``query``; returns a Struct{index, score} array."""
    _require_arrow_embeddings("arrow_embeddings_rerank")
    _require_real("arrow_embeddings_rerank", query=query, candidates=candidates)
    return _raw.arrow_embeddings_rerank(query, candidates, k, metric)


def arrow_embeddings_normalize_rows(rows):
    """Normalize each row to unit L2 length, returning a FixedSizeList matrix."""
    _require_arrow_embeddings("arrow_embeddings_normalize_rows")
    _require_real("arrow_embeddings_normalize_rows", rows=rows)
    return _raw.arrow_embeddings_normalize_rows(rows)


def arrow_embeddings_brute_force_knn(queries, corpus, k, metric="cosine"):
    """Best ``k`` corpus neighbors per query row; returns a List of Struct{index, score} arrays."""
    _require_arrow_embeddings("arrow_embeddings_brute_force_knn")
    _require_real("arrow_embeddings_brute_force_knn", queries=queries, corpus=corpus)
    return _raw.arrow_embeddings_brute_force_knn(queries, corpus, k, metric)


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
    "arrow_embeddings_brute_force_knn",
    "arrow_embeddings_normalize_rows",
    "arrow_embeddings_query_corpus_scores",
    "arrow_embeddings_rerank",
    "arrow_fixed_shape_tensor_array",
    "arrow_fixed_shape_tensor_numpy",
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
    "arrow_tensor_batched_dot_last_axis",
    "arrow_tensor_batched_matmul_last_two",
    "arrow_tensor_contract_axes",
    "arrow_tensor_cp_als3",
    "arrow_tensor_cp_als3_diagnostics",
    "arrow_tensor_cp_als3_reconstruct",
    "arrow_tensor_cp_als3_with_report",
    "arrow_tensor_cp_als_nd",
    "arrow_tensor_cp_als_nd_diagnostics",
    "arrow_tensor_cp_als_nd_reconstruct",
    "arrow_tensor_cp_als_nd_with_report",
    "arrow_tensor_cube_matmat",
    "arrow_tensor_cube_matvec",
    "arrow_tensor_einsum",
    "arrow_tensor_flatten_cubes",
    "arrow_tensor_hooi_nd",
    "arrow_tensor_hosvd_nd",
    "arrow_tensor_hosvd_nd_reconstruct",
    "arrow_tensor_l2_norm_last_axis",
    "arrow_tensor_normalize_last_axis",
    "arrow_tensor_permute_axes",
    "arrow_tensor_sum_last_axis",
    "arrow_tensor_tt_add",
    "arrow_tensor_tt_hadamard",
    "arrow_tensor_tt_hadamard_round",
    "arrow_tensor_tt_inner",
    "arrow_tensor_tt_norm",
    "arrow_tensor_tt_orthogonalize_left",
    "arrow_tensor_tt_orthogonalize_right",
    "arrow_tensor_tt_round",
    "arrow_tensor_tt_svd",
    "arrow_tensor_tt_svd_reconstruct",
    "arrow_tensor_tucker_expand",
    "arrow_tensor_tucker_project",
    "arrow_variable_shape_tensor_array",
    "arrow_variable_shape_tensor_rows",
]
