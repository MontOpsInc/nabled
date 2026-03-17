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
nabled-ml = "0.0.7"
```

## Optional Features

1. `blas`: forwards BLAS support into `nabled-linalg`.
2. `openblas-system`: enables provider-backed LAPACK paths via system OpenBLAS.
3. `openblas-static`: enables provider-backed LAPACK paths via statically linked OpenBLAS.
4. `netlib-system`: enables provider-backed LAPACK paths via system Netlib LAPACK.
5. `netlib-static`: enables provider-backed LAPACK paths via statically linked Netlib LAPACK.
6. `magma-system`: enables NVIDIA MAGMA provider-backed decomposition paths where used through `nabled-linalg`.

Note: provider requirements depend on provider choice. Static features (`openblas-static`,
`netlib-static`) require native build toolchains such as `gcc`, `gfortran`, and `make`, and
`netlib-system` requires a system `LAPACK`/Fortran runtime available to the linker.

```toml
[dependencies]
nabled-ml = { version = "0.0.7", features = ["openblas-system"] }
```

## Docs

1. API docs: <https://docs.rs/nabled-ml>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
