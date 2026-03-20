# pynabled

Python bindings for **[nabled](https://github.com/MontOpsInc/nabled)** — ndarray-style linear algebra and ML-oriented routines (SVD, QR, LU, Cholesky, eigen, PCA, sparse solvers, tensors, and more).

## Install from PyPI

```bash
pip install pynabled
```

Requires Python 3.10+ and NumPy.

## Install from source (development)

```bash
pip install maturin numpy
maturin develop
```

Optional Cargo features (OpenBLAS, `accelerator-rayon`, Arrow interop) require explicit build flags; see **[BUILD.md](../BUILD.md)** in the repository root.

## Quick example

```python
import numpy as np
import pynabled

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="C")
u, s, vt = pynabled.svd_decompose(a)
print("singular values:", s)
```

Use **C-contiguous** `float64` arrays for zero-copy paths; otherwise call `np.ascontiguousarray(a)` first.

## Documentation

- Rust library API: [docs.rs/nabled](https://docs.rs/nabled)
- Full Python surface: `python/pynabled/__init__.py`
- Examples: `python/examples/`
- Publishing wheels to PyPI: [docs/PYPI_PUBLISH.md](https://github.com/MontOpsInc/nabled/blob/main/docs/PYPI_PUBLISH.md) (also in-repo under `docs/`)

## License

MIT OR Apache-2.0 (same as the nabled workspace).
