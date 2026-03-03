# nabled-ml

ML-oriented numerical algorithms built on ndarray-native `nabled` primitives.

`nabled-ml` includes:

1. Iterative solvers.
2. Optimization routines.
3. Numerical Jacobian/gradient/Hessian helpers.
4. PCA and regression.
5. Statistical utilities.

## Install

```toml
[dependencies]
nabled-ml = "0.0.1"
```

## Optional Features

1. `blas`: forwards BLAS support into `nabled-linalg`.
2. `openblas-system`: enables provider-backed LAPACK paths through `nabled-linalg`.

```toml
[dependencies]
nabled-ml = { version = "0.0.1", features = ["openblas-system"] }
```

## Docs

1. API docs: <https://docs.rs/nabled-ml>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
