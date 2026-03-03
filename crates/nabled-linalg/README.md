# nabled-linalg

Ndarray-native linear algebra domains for `nabled`.

`nabled-linalg` includes dense/sparse decomposition routines, matrix/vector/tensor
primitives, and matrix-function algorithms.

## Install

```toml
[dependencies]
nabled-linalg = "0.0.1"
```

## Optional Features

1. `blas`: enables `ndarray/blas`.
2. `openblas-system`: enables provider-backed LAPACK paths via OpenBLAS.
3. `accelerator-rayon`: enables selected parallel CPU kernels.
4. `accelerator-wgpu`: enables bounded WGPU-backed kernel paths.

```toml
[dependencies]
nabled-linalg = { version = "0.0.1", features = ["openblas-system"] }
```

## Docs

1. API docs: <https://docs.rs/nabled-linalg>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
