# ꩜ Nabled - Rust Linear Algebra

[![Crates.io](https://img.shields.io/crates/v/nabled.svg)](https://crates.io/crates/nabled)
[![Documentation](https://docs.rs/nabled/badge.svg)](https://docs.rs/nabled)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/MontOpsInc/nabled/ci.yml?branch=main)](https://github.com/MontOpsInc/nabled/actions)
[![Coverage](https://codecov.io/gh/MontOpsInc/nabled/branch/main/graph/badge.svg)](https://codecov.io/gh/MontOpsInc/nabled)

Nabled is a linear algebra library in Rust with dual `nalgebra` and `ndarray` backends.

## Project Status

- Pre-release, private repository, API still evolving.
- Current package name in `Cargo.toml` is `rust-linalg`.
- Numerical robustness, testing, and documentation are active priorities.

## What Is Implemented

- Matrix decompositions: SVD, QR, LU, Cholesky, Schur, Polar, Eigen
- Solvers: least squares, triangular solve, Sylvester/Lyapunov, iterative (CG/GMRES)
- Higher-level methods: PCA and linear regression
- Matrix functions: exponential, logarithm, and fractional/integer power
- Numerical differentiation: Jacobian, gradient, Hessian (real + complex)
- Statistics: column means, centering, covariance, correlation

Most modules expose `nalgebra_*` and `ndarray_*` submodules so both backends share the same conceptual API.

## Installation

This crate is not published yet (`publish = false`).

Use a path dependency while developing locally:

```toml
[dependencies]
rust-linalg = { path = "../nabled" }
```

Or a Git dependency:

```toml
[dependencies]
rust-linalg = { git = "https://github.com/MontOpsInc/nabled" }
```

## Quick Start

### SVD

```rust
use nalgebra::DMatrix;
use rust_linalg::svd::nalgebra_svd;

fn main() -> Result<(), rust_linalg::svd::SVDError> {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let svd = nalgebra_svd::compute_svd(&a)?;

    println!("Singular values: {:?}", svd.singular_values);
    println!("Condition number: {}", nalgebra_svd::condition_number(&svd));
    Ok(())
}
```

### QR Least Squares

```rust
use nalgebra::{DMatrix, DVector};
use rust_linalg::qr::{QRConfig, nalgebra_qr};

fn main() -> Result<(), rust_linalg::qr::QRError> {
    let a = DMatrix::from_row_slice(4, 2, &[
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
    ]);
    let b = DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

    let x = nalgebra_qr::solve_least_squares(&a, &b, &QRConfig::default())?;
    println!("x = {x}");
    Ok(())
}
```

### Numerical Jacobian

```rust
use nalgebra::DVector;
use rust_linalg::jacobian::{JacobianConfig, nalgebra_jacobian};

fn main() -> Result<(), rust_linalg::jacobian::JacobianError> {
    let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
        Ok(DVector::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
    };

    let x = DVector::from_vec(vec![2.0, 3.0]);
    let j = nalgebra_jacobian::numerical_jacobian(&f, &x, &JacobianConfig::default())?;
    println!("J = {j}");
    Ok(())
}
```

## Module Guide

- `svd`: full/truncated SVD, rank/condition number, pseudo-inverse, null space
- `qr`: full/reduced/pivoted QR, least-squares, reconstruction, condition number
- `lu`: LU decomposition, solve, inverse, log-determinant
- `cholesky`: SPD decomposition, solve, inverse
- `eigen`: symmetric and generalized eigen problems
- `matrix_functions`: exp/log/power operations
- `jacobian`: numerical Jacobian/gradient/Hessian (real and complex)
- `pca`: PCA fit, transform, inverse transform
- `regression`: ordinary least squares
- `schur`: Schur decomposition
- `sylvester`: Sylvester and Lyapunov equation solvers
- `iterative`: iterative linear solvers
- `stats`: means, centering, covariance, correlation
- `triangular`: forward/back substitution utilities

## Examples

```bash
cargo run --example svd_example
cargo run --example qr_example
cargo run --example jacobian_example
cargo run --example matrix_functions_example
cargo run --example pca_example
cargo run --example regression_example
```

## Quality Gates

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -W clippy::pedantic
cargo test --all-targets --all-features
cargo test --test integration -F test-utils
```

## Benchmarks

```bash
cargo bench
```

## License

Licensed under either:

- MIT license
- Apache License, Version 2.0

at your option.
