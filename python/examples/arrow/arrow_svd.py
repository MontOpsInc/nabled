#!/usr/bin/env python3
"""SVD from PyArrow: zero-copy SVD on Arrow FixedSizeListArray matrix.

Demonstrates the PyArrow bridge: create a matrix as a PyArrow FixedSizeListArray
(rows as lists), run SVD via pynabled.arrow, and verify round-trip.

Requires: numpy, pyarrow, pynabled built with Arrow support

Build: maturin develop
Run from repo root:
    python python/examples/arrow/arrow_svd.py
"""

import numpy as np
import pyarrow as pa

from pynabled.arrow import arrow_svd_decompose


def main():
    # Create a 4x3 matrix as FixedSizeListArray (each row is a list of 3 floats)
    data = pa.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        type=pa.list_(pa.float64(), 3),
    )

    u, s, vt = arrow_svd_decompose(data)
    u = np.asarray(u)
    s = np.asarray(s)
    vt = np.asarray(vt)

    # Reconstruct and compare to original
    a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )
    recon = u @ np.diag(s) @ vt
    err = np.linalg.norm(a - recon, "fro")
    print("Arrow SVD (pynabled.arrow)")
    print("-" * 40)
    print(f"Matrix shape: {a.shape}")
    print(f"Singular values: {s}")
    print(f"Reconstruction Frobenius error: {err:.2e}")
    assert err < 1e-10, "Round-trip reconstruction failed"
    print("OK: Round-trip verified.")


if __name__ == "__main__":
    main()
