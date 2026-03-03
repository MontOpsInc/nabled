# Nabled

[![Crates.io](https://img.shields.io/crates/v/nabled.svg)](https://crates.io/crates/nabled)
[![Documentation](https://docs.rs/nabled/badge.svg)](https://docs.rs/nabled)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/MontOpsInc/nabled/ci.yml?branch=main)](https://github.com/MontOpsInc/nabled/actions)
[![Coverage](https://codecov.io/gh/MontOpsInc/nabled/branch/main/graph/badge.svg)](https://codecov.io/gh/MontOpsInc/nabled)

Nabled is an ndarray-native Rust numerical library focused on production-grade
linear algebra and ML-oriented matrix/vector operations.

## Install

```toml
[dependencies]
nabled = "0.0.1"
```

## Current Direction

1. Ndarray-native API and implementation model.
2. Strict focus on correctness, performance, and composability.
3. No Arrow-specific types in `nabled` APIs.
4. Workspace architecture (`nabled-core`, `nabled-linalg`, `nabled-ml`, facade `nabled`).

See [`docs/README.md`](docs/README.md) for current architecture and roadmap.

## Implemented Domains

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

## Namespaced API

1. `nabled::core`: shared errors, validation, and prelude exports.
2. `nabled::linalg`: linear algebra and decomposition modules.
3. `nabled::ml`: ML-oriented numerical routines.

## Features

1. `blas`: enables `ndarray/blas` across participating workspace crates.
2. `openblas-system`: enables provider-backed OpenBLAS/LAPACK paths.
3. `accelerator-rayon`: enables selected parallel CPU kernels.
4. `accelerator-wgpu`: enables selected WGPU-backed kernels.

```toml
[dependencies]
nabled = { version = "0.0.1", features = ["openblas-system"] }
```

Feature behavior:

1. `openblas-system` implies `blas`.
2. Provider selection is compile-time and internal to decomposition paths.
3. Backend acceleration is compile-time and kernel-family-specific.

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
