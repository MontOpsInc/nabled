"""PyArrow bridge for nabled/ndarrow workflows.

Requires pynabled built with the arrow feature: pip install pynabled[arrow]
and maturin develop --features arrow (or equivalent).
"""

from .results import SvdResult

try:
    from pynabled._pynabled import (
        arrow_dot,
        arrow_l2_norm,
    )
    from pynabled._pynabled import (
        arrow_svd_decompose as _arrow_svd_decompose,
    )
except ImportError as e:
    raise ImportError(
        "pynabled arrow support not available. "
        "Install with: pip install pynabled[arrow] and build with --features arrow"
    ) from e


def arrow_svd_decompose(data) -> SvdResult:
    u, singular_values, vt = _arrow_svd_decompose(data)
    return SvdResult(u=u, singular_values=singular_values, vt=vt)


__all__ = [
    "arrow_dot",
    "arrow_l2_norm",
    "arrow_svd_decompose",
]
