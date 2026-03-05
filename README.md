# 🕸️ Nabled

[![Crates.io](https://img.shields.io/crates/v/nabled.svg)](https://crates.io/crates/nabled)
[![Documentation](https://docs.rs/nabled/badge.svg)](https://docs.rs/nabled)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/MontOpsInc/nabled/ci.yml?branch=main)](https://github.com/MontOpsInc/nabled/actions)
[![Coverage](https://codecov.io/gh/MontOpsInc/nabled/branch/main/graph/badge.svg)](https://codecov.io/gh/MontOpsInc/nabled)

Nabled is an ndarray-native Rust numerical library focused on production-grade
linear algebra and ML-oriented matrix/vector operations.

Important! Nabled is under active development right now, so the only way to be sure the public APIs don't break is to pin your version. When stabilizied, it will follow proper versioning, but for now it is guaranteed to change. 

## Install

```toml
[dependencies]
nabled = "0.0.4"
```

## Implemented Domains

This list is ever-changing, consult the Rust Docs for the source of truth. 

1. SVD, QR, LU, Cholesky, Eigen, Schur, Polar
2. Matrix functions (exp/log/power/sign)
3. Triangular solve, Sylvester/Lyapunov
4. PCA, regression, iterative solvers
5. Numerical Jacobian/gradient/Hessian
6. Statistics utilities
7. Vector primitives (dot/norm/cosine/pairwise/batched)

## Quick Example

```rust
use ndarray::arr2;
use nabled::linalg::svd;

fn main() -> Result<(), nabled::linalg::svd::SVDError> {
    let a = arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]);
    let svd = svd::decompose(&a)?;
    println!("singular values = {:?}", svd.singular_values);
    Ok(())
}
```

Review more examples in `crates/nabled/examples`.

## Namespaced API

1. `nabled::core`: shared errors, validation, and prelude exports.
2. `nabled::linalg`: linear algebra and decomposition modules.
3. `nabled::ml`: ML-oriented numerical routines.

## Features

1. `blas`: enables `ndarray/blas` across participating workspace crates.
2. `openblas-system`: enables provider-backed LAPACK paths via system OpenBLAS.
3. `openblas-static`: enables provider-backed LAPACK paths via statically linked OpenBLAS.
4. `netlib-system`: enables provider-backed LAPACK paths via system Netlib LAPACK.
5. `netlib-static`: enables provider-backed LAPACK paths via statically linked Netlib LAPACK.
6. `accelerator-rayon`: enables selected parallel CPU kernels.
7. `accelerator-wgpu`: enables WGPU-backed `f32` kernel paths across dense/vector/tensor families.

```toml
[dependencies]
nabled = { version = "0.0.4", features = ["openblas-system"] }
```

Feature behavior:

1. `openblas-system` implies `blas`.
2. Provider feature selection (`openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`) is compile-time and internal to decomposition paths.
3. Backend acceleration is compile-time and kernel-family-specific.
4. GPU-backend-dispatched `f32` kernels use explicit CPU fallback when no usable GPU is available.
5. Provider/toolchain requirements depend on backend choice; `openblas-static` and
   `netlib-static` require native build toolchains (`gcc`/`gfortran`/`make`), and `netlib-system`
   requires a system `LAPACK`/Fortran runtime available to the linker.

## Quality Gates

```bash
just checks
```

On macOS, provider-enabled `just` recipes automatically set `PKG_CONFIG_PATH` and `OPENBLAS_DIR` for Homebrew OpenBLAS (`/opt/homebrew/opt/openblas`). No manual env export is required for those recipes.

## Benchmarks

```bash
just bench-smoke-report
```

## License

Licensed under either:

1. MIT license
2. Apache License, Version 2.0

at your option.
