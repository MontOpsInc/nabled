# Nabled

[![Crates.io](https://img.shields.io/crates/v/nabled.svg)](https://crates.io/crates/nabled)
[![Documentation](https://docs.rs/nabled/badge.svg)](https://docs.rs/nabled)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/MontOpsInc/nabled/ci.yml?branch=main)](https://github.com/MontOpsInc/nabled/actions)
[![Coverage](https://codecov.io/gh/MontOpsInc/nabled/branch/main/graph/badge.svg)](https://codecov.io/gh/MontOpsInc/nabled)

Nabled is a Rust numerical library focused on high-performance linear algebra and ML-oriented matrix/vector operations over ndarray data structures.

## Current Direction

1. Ndarray-first API and implementation model.
2. Strict focus on correctness, performance, and composability.
3. No Arrow-specific types in `nabled` APIs.
4. Workspace migration in progress to support long-term scale.

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
use nabled::svd::ndarray_svd;

fn main() -> Result<(), nabled::svd::SVDError> {
    let a = arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]);
    let svd = ndarray_svd::decompose(&a)?;
    println!("singular values = {:?}", svd.singular_values);
    Ok(())
}
```

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
