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
result = pynabled.svd_decompose(a)
print("singular values:", result.singular_values)
```

`numpy.ndarray` is the canonical CPU-array carrier for `pynabled`. Borrowed NumPy views are used
where the Rust API admits them, including non-C-contiguous inputs for the view-based dense paths.
Current real-valued vector/matrix/decomposition/function/batched/statistics/regression/PCA/
iterative/callable-ML/tensor bindings all accept both `float32` and `float64` under the same
public function names. The currently exposed Arrow rows (`arrow_dot`, `arrow_l2_norm`,
`arrow_svd_decompose`) also accept both real dtypes, and current Arrow SVD NumPy egress preserves
the caller's real dtype. Mixed real dtypes are rejected explicitly instead of being silently cast.
Callable-driven `jacobian` / `optimization` workflows preserve the caller's real dtype end-to-end,
and their `float32` defaults use `float32`-appropriate finite-difference / convergence settings
instead of raw `float64` thresholds. Some higher-level wrappers still materialize owned arrays
internally where the current Rust API shape requires it.

Where the admitted Rust surface already has direct complex kernels, the current Python dense API
now also accepts `complex128` across the main vector/matrix/decomposition/function families
(`dot`, `l2_norm`, `cosine_similarity`, `matvec`, `matmat`, `svd`, `qr`, `lu` solve/inverse/
determinant, `cholesky`, non-symmetric `eigen`, `schur`, `polar`, `sylvester`, `lyapunov`,
admitted complex `matrix_functions`, and `gram_schmidt`). Unsupported rows fail explicitly rather
than silently casting or dropping back to a different numerical contract.

Structured decomposition / ML / tensor workflows now return typed Python result objects with named
fields instead of anonymous tuples. For example, `svd_decompose(...)` returns `pynabled.SvdResult`,
`compute_pca(...)` returns `pynabled.PcaResult`, and tensor decomposition helpers return
corresponding `Hosvd*` / `CpAls*` / `TensorTrainResult` objects.

Sparse CSR workflows use `pynabled.CsrMatrix` as the canonical Python carrier. SciPy-compatible
objects can be normalized into that carrier explicitly with `CsrMatrix.from_scipy(...)` or passed
to the public sparse wrappers directly. The current CSR carrier preserves `int32` / `int64` index
buffers and `float32` / `float64` data rather than normalizing everything to one dtype. Explicit
normalization is available through `dtype=` / `index_dtype=` on construction plus
`CsrMatrix.astype(...)` and `CsrMatrix.with_index_dtype(...)`, while mixed sparse operand dtypes
continue to fail explicitly instead of being silently cast.

Sparse reuse paths are now first-class as well: `CsrMatrix` can build reusable
`JacobiPreconditioner`, `ILU0` / `ILUT` / `ILUK`, `IC0`, `ILDL0`, and direct
`SparseLUFactorization` objects. Those wrappers keep the reusable sparse state in Rust instead of
reconstructing factors from NumPy on every apply/solve call, while factor properties still round
trip back through canonical `CsrMatrix` carriers. `ILUTConfig` and `ILUKConfig` expose the current
configurable sparse setup profiles, and sparse LU reuse already covers both single-RHS and
multi-RHS solve workflows.

Sparse carrier breadth is now wider as well: `pynabled` also exposes first-class `CscMatrix` and
`CooMatrix` carriers with SciPy-compatible ingress plus explicit `CSR -> CSC`, `CSC -> CSR`, and
`COO -> CSR` conversions. CSC matvec stays native to the CSC carrier, sparse-sparse matmat returns
the canonical `CsrMatrix`, and the reusable `ILU0` / `ILUT` / `ILUK` / `ILDL0` factor objects now
also drive GMRES / `BiCGSTAB` solve and multi-RHS solve workflows without rebuilding the sparse
factorization each call. Where a sparse kernel only admits the CSR form today, `pynabled` keeps
that normalization explicit through the carrier methods rather than silently pretending the storage
format is interchangeable.

## Documentation

- Rust library API: [docs.rs/nabled](https://docs.rs/nabled)
- Full Python surface: `python/pynabled/__init__.py`
- Examples: `python/examples/`
- Publishing wheels to PyPI: [docs/PYPI_PUBLISH.md](https://github.com/MontOpsInc/nabled/blob/main/docs/PYPI_PUBLISH.md) (also in-repo under `docs/`)

## License

MIT OR Apache-2.0 (same as the nabled workspace).
