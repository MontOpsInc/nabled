# nabled-linalg

Ndarray-native linear algebra domains for `nabled`.

`nabled-linalg` includes dense/sparse decomposition routines, matrix/vector/tensor
primitives, and matrix-function algorithms.

## Install

```toml
[dependencies]
nabled-linalg = "0.0.4"
```

## Optional Features

1. `blas`: enables `ndarray/blas`.
2. `openblas-system`: enables provider-backed LAPACK paths via system OpenBLAS.
3. `openblas-static`: enables provider-backed LAPACK paths via statically linked OpenBLAS.
4. `netlib-system`: enables provider-backed LAPACK paths via system Netlib LAPACK.
5. `netlib-static`: enables provider-backed LAPACK paths via statically linked Netlib LAPACK.
6. `accelerator-rayon`: enables selected parallel CPU kernels.
7. `accelerator-wgpu`: enables bounded WGPU-backed dense/vector/tensor kernel paths (`f32` native, `f64` native when `SHADER_F64` is available).

Note: provider requirements depend on backend choice. Static features (`openblas-static`,
`netlib-static`) require native build toolchains such as `gcc`, `gfortran`, and `make`, and
`netlib-system` requires a system `LAPACK`/Fortran runtime available to the linker.

```toml
[dependencies]
nabled-linalg = { version = "0.0.4", features = ["openblas-system"] }
```

## Docs

1. API docs: <https://docs.rs/nabled-linalg>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
