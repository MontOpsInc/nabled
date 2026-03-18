"""Tests for PyArrow bridge (requires pynabled built with --features arrow)."""

import numpy as np
import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    from pynabled.arrow import arrow_dot, arrow_l2_norm, arrow_svd_decompose
except ImportError:
    arrow_dot = None
    arrow_l2_norm = None
    arrow_svd_decompose = None


pytestmark = [
    pytest.mark.skipif(pa is None, reason="pyarrow not installed"),
    pytest.mark.skipif(arrow_dot is None, reason="pynabled built without arrow feature"),
]


def test_arrow_dot():
    a = pa.array([1.0, 2.0, 3.0])
    b = pa.array([4.0, 5.0, 6.0])
    result = arrow_dot(a, b)
    expected = 1 * 4 + 2 * 5 + 3 * 6
    assert abs(result - expected) < 1e-10


def test_arrow_l2_norm():
    a = pa.array([3.0, 4.0])
    result = arrow_l2_norm(a)
    assert abs(result - 5.0) < 1e-10


def test_arrow_svd_decompose():
    # Create a 2x2 matrix as FixedSizeListArray (rows as lists)
    data = pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float64(), 2))
    u, s, vt = arrow_svd_decompose(data)
    u, s, vt = np.asarray(u), np.asarray(s), np.asarray(vt)
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    recon = u @ np.diag(s) @ vt
    np.testing.assert_allclose(recon, a, rtol=1e-10)
