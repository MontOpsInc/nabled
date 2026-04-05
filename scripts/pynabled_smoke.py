#!/usr/bin/env python3
"""Exercise a small but real pynabled API slice from an installed package."""

from __future__ import annotations

import argparse

import numpy as np
import pynabled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-arrow", action="store_true")
    args = parser.parse_args()

    left = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    right = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    assert abs(pynabled.dot(left, right) - 32.0) < 1e-10

    matrix = pynabled.CsrMatrix.from_components(
        (3, 3),
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )
    matvec = matrix @ left
    np.testing.assert_allclose(matvec, np.array([1.0, 4.0, 9.0], dtype=np.float64), rtol=1e-10)

    if args.require_arrow:
        import pyarrow as pa
        from pynabled.arrow import arrow_dot

        arrow_result = arrow_dot(pa.array([1.0, 2.0]), pa.array([3.0, 4.0]))
        assert abs(arrow_result - 11.0) < 1e-10


if __name__ == "__main__":
    main()
