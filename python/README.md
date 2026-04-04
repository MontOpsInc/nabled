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

`numpy.ndarray` is the canonical CPU-array carrier for `pynabled`. Borrowed NumPy views are used
where the Rust API admits them, including non-C-contiguous inputs for the view-based dense paths.
Current real-valued vector/matrix/statistics/regression/PCA/iterative bindings accept both
`float32` and `float64` under the same public function names. Mixed real dtypes are rejected
explicitly instead of being silently cast. Some higher-level wrappers still materialize owned
arrays internally where the current Rust API shape requires it.

Sparse CSR workflows use `pynabled.CsrMatrix` as the canonical Python carrier. SciPy-compatible
objects can be normalized into that carrier explicitly with `CsrMatrix.from_scipy(...)` or passed
to the public sparse wrappers directly. The current CSR carrier preserves `int32` / `int64` index
buffers and `float32` / `float64` data rather than normalizing everything to one dtype. Explicit
normalization is available through `dtype=` / `index_dtype=` on construction plus
`CsrMatrix.astype(...)` and `CsrMatrix.with_index_dtype(...)`, while mixed sparse operand dtypes
continue to fail explicitly instead of being silently cast.

## Documentation

- Rust library API: [docs.rs/nabled](https://docs.rs/nabled)
- Full Python surface: `python/pynabled/__init__.py`
- Examples: `python/examples/`
- Publishing wheels to PyPI: [docs/PYPI_PUBLISH.md](https://github.com/MontOpsInc/nabled/blob/main/docs/PYPI_PUBLISH.md) (also in-repo under `docs/`)

## License

MIT OR Apache-2.0 (same as the nabled workspace).
