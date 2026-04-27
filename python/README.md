# pynabled

`pynabled` is the Python package for `nabled`, an ndarray-native numerical library for dense
linear algebra, decompositions, sparse workflows, tensor routines, and selected ML/statistics
operations.

## Install

Install the published wheel:

```bash
pip install pynabled
```

`pynabled` requires Python 3.10+ and NumPy.

## Optional features

Published PyPI wheels use the default feature set.

Default wheels include the Rust `arrow` feature. Python Arrow workflows still require `pyarrow`
at runtime.

Optional provider and accelerator support are source-build workflows that use the same Cargo
feature names as the Rust facade:

- `openblas-system`
- `openblas-static`
- `netlib-system`
- `netlib-static`
- `magma-system`
- `accelerator-rayon`
- `accelerator-wgpu`

For `pip` / `uv` source builds, the package exposes friendly build settings:

```bash
PYNABLED_PROVIDER="openblas-system magma-system" python -m pip install --no-binary pynabled pynabled
PYNABLED_ACCELERATORS=rayon python -m pip install --no-binary pynabled pynabled
```

You can pass the same knobs through frontend config settings with
`pynabled-provider`, `pynabled-accelerators`, and `pynabled-features`.

Builds can be inspected at runtime with `pynabled.build_features()`.

Build guide:
<https://github.com/MontOpsInc/nabled/blob/main/BUILD.md>

## Quick example

```python
import numpy as np
import pynabled

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
result = pynabled.svd_decompose(a)

print(result.singular_values)
```

## What it includes

- Dense vector and matrix kernels over `numpy.ndarray`
- Decompositions including SVD, QR, LU, Cholesky, eigen, Schur, and polar
- Sparse matrix carriers and solver/preconditioner workflows
- Tensor decomposition and reconstruction helpers
- PCA, regression, iterative solvers, Jacobian helpers, and optimization configs
- Optional Arrow and `ndarrow` interop when built with `arrow`

## API and behavior

`numpy.ndarray` is the canonical CPU array carrier. Where the Rust API admits borrowed views,
`pynabled` preserves that contract at the Python boundary instead of forcing extra copies.

Structured results are returned as typed Python objects such as `SvdResult`, `QrResult`,
`LuResult`, `PcaResult`, and tensor-specific result types.

Some higher-level convenience APIs still materialize owned arrays internally where the current
Rust API shape requires it. Callback-driven Jacobian and optimization helpers also cross back into
Python on each evaluation, so they do not have the same performance contract as the direct
array-in/array-out kernels.

## Project links

- Repository: <https://github.com/MontOpsInc/nabled>
- Python package docs: <https://github.com/MontOpsInc/nabled/blob/main/python/README.md>
- Build guide: <https://github.com/MontOpsInc/nabled/blob/main/BUILD.md>
- Changelog: <https://github.com/MontOpsInc/nabled/blob/main/CHANGELOG.md>
