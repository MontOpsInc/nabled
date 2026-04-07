"""PyArrow bridge for nabled/ndarrow workflows.

Requires pynabled built with the arrow feature: ``pip install pynabled[arrow]``
and ``maturin develop --features arrow`` (or equivalent).
"""

import pyarrow as pa

from .results import SvdResult

try:
    from pynabled._pynabled import (
        arrow_batched_cosine_distance,
        arrow_batched_cosine_similarity,
        arrow_batched_dot,
        arrow_batched_l2_norm,
        arrow_batched_normalize,
        arrow_batched_row_matvec,
        arrow_center_columns,
        arrow_column_means,
        arrow_correlation_matrix,
        arrow_cosine_distance,
        arrow_cosine_similarity,
        arrow_covariance_matrix,
        arrow_dot,
        arrow_l2_norm,
        arrow_matmat,
        arrow_matvec,
        arrow_pairwise_cosine_distance,
        arrow_pairwise_cosine_similarity,
        arrow_pairwise_l2_distance,
    )
    from pynabled._pynabled import (
        arrow_batched_matmat as _arrow_batched_matmat,
    )
    from pynabled._pynabled import (
        arrow_batched_matmat_broadcast_left as _arrow_batched_matmat_broadcast_left,
    )
    from pynabled._pynabled import (
        arrow_batched_matmat_broadcast_right as _arrow_batched_matmat_broadcast_right,
    )
    from pynabled._pynabled import (
        arrow_svd_decompose as _arrow_svd_decompose,
    )
except ImportError as e:
    raise ImportError(
        "pynabled arrow support not available. "
        "Install with: pip install pynabled[arrow] and build with --features arrow"
    ) from e


def _arrow_field(array, name):
    return pa.field(name, array.type, nullable=False)


def _extension_array(field, storage):
    return pa.ExtensionArray.from_storage(field.type, storage)


def arrow_batched_matmat(left, right):
    field, storage = _arrow_batched_matmat(
        _arrow_field(left, "left"),
        left,
        _arrow_field(right, "right"),
        right,
    )
    return _extension_array(field, storage)


def arrow_batched_matmat_broadcast_right(left, right):
    field, storage = _arrow_batched_matmat_broadcast_right(_arrow_field(left, "left"), left, right)
    return _extension_array(field, storage)


def arrow_batched_matmat_broadcast_left(left, right):
    field, storage = _arrow_batched_matmat_broadcast_left(left, _arrow_field(right, "right"), right)
    return _extension_array(field, storage)


def arrow_svd_decompose(data) -> SvdResult:
    u, singular_values, vt = _arrow_svd_decompose(data)
    return SvdResult(u=u, singular_values=singular_values, vt=vt)


__all__ = [
    "arrow_batched_cosine_distance",
    "arrow_batched_cosine_similarity",
    "arrow_batched_dot",
    "arrow_batched_l2_norm",
    "arrow_batched_matmat",
    "arrow_batched_matmat_broadcast_left",
    "arrow_batched_matmat_broadcast_right",
    "arrow_batched_normalize",
    "arrow_batched_row_matvec",
    "arrow_center_columns",
    "arrow_column_means",
    "arrow_correlation_matrix",
    "arrow_cosine_distance",
    "arrow_cosine_similarity",
    "arrow_covariance_matrix",
    "arrow_dot",
    "arrow_l2_norm",
    "arrow_matmat",
    "arrow_matvec",
    "arrow_pairwise_cosine_distance",
    "arrow_pairwise_cosine_similarity",
    "arrow_pairwise_l2_distance",
    "arrow_svd_decompose",
]
