# 🕸️ Nabled

[![Crates.io](https://img.shields.io/crates/v/nabled.svg)](https://crates.io/crates/nabled)
[![Documentation](https://docs.rs/nabled/badge.svg)](https://docs.rs/nabled)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/MontOpsInc/nabled/ci.yml?branch=main)](https://github.com/MontOpsInc/nabled/actions)
[![Coverage](https://codecov.io/gh/MontOpsInc/nabled/branch/main/graph/badge.svg)](https://codecov.io/gh/MontOpsInc/nabled)

Nabled is an ndarray-native Rust numerical library focused on production-grade
linear algebra and ML-oriented matrix/vector operations.

Optional Arrow interop is available behind feature `arrow`, using
[ndarrow](https://crates.io/crates/ndarrow) as the zero-copy Arrow/ndarray bridge while keeping
the core numerical crates ndarray-native.
When feature `arrow` is enabled, `nabled` also re-exports that bridge as `nabled::ndarrow` so
Arrow-facing consumers can stay on the same bridge contract version as the facade adapters.

Important! Nabled is under active development right now, so the only way to be sure the public APIs don't break is to pin your version. When stabilized, it will follow proper versioning, but for now it is guaranteed to change.

## Install

```toml
[dependencies]
nabled = "0.0.8"
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

## Python

Python bindings are available via the `pynabled` package. Install with [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin numpy
maturin develop
```

Then use nabled from Python with NumPy arrays:

```python
import numpy as np
import pynabled

a = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
result = pynabled.svd_decompose(a)
print("singular values:", result.singular_values)
```

The package exposes SVD, QR, LU, Cholesky, eigen, Schur, polar, Sylvester/Lyapunov, triangular solve, matrix functions, orthogonalization, dense vector/matrix primitives (including batched vector helpers and broadcasted batched matmat), batched decompositions, tensor ops, regression, PCA, statistics, and a widened sparse surface with first-class CSR/CSC/COO carriers, direct sparse iterative solvers, reusable sparse factorization/preconditioner workflows, and direct ILU(0)/ILUT/ILUK/ILDL0 GMRES / `BiCGSTAB` convenience rows. The Arrow-facing `pynabled.arrow` module now also covers the admitted real dense/decomposition slice plus canonical complex dense/vector/matrix/statistics/orthogonalization/triangular/decomposition/matrix-function/PCA/regression rows, typed batched QR/SVD/LU/Cholesky/symmetric-eigen results over PyArrow fixed-shape tensors, callback-driven iterative/Jacobian/optimization rows, canonical sparse CSR object/batch carriers with direct sparse solve/product/reuse workflows, and canonical fixed-shape / variable-shape tensor workflows across last-axis ops, permutation/contraction, batched matmul, cube kernels, einsum, CP-ALS, HOSVD/HOOI/Tucker, and TT helpers over PyArrow/`ndarrow`. Arrow-native outputs stay Arrow-native where the Rust Arrow facade already defines them, while ndarray-native decomposition/PCA/regression/tensor results are exposed through the same typed Python result objects used by the NumPy-facing API. See `python/pynabled/__init__.py` and `python/pynabled/arrow.py` for the full API surface.

Dense iterative plus callable-driven Jacobian/optimization rows now also expose typed Python
config objects (`IterativeConfig`, `JacobianConfig`, `LineSearchConfig`, `GradientDescentConfig`,
`AdamConfig`, `MomentumConfig`, `RMSPropConfig`, `ProjectedGradientConfig`, `BFGSConfig`) instead
of treating raw tuning-parameter shims as the long-term contract. The callback-driven Jacobian and
optimizer helpers remain convenience APIs rather than no-compromise hot-path equivalents, because
their objective/gradient evaluations still cross back into Python.

The Python NumPy-facing API now also exposes explicit output-buffer reuse beyond the primitive
vector/matrix/tensor kernels: `svd_pseudo_inverse`, `svd_reconstruct_matrix`, `matrix_exp`,
`matrix_log_taylor`, `matrix_log_eigen`, `matrix_log_svd`, `matrix_power`, `matrix_sign`,
`matrix_exp_eigen`, `sylvester_solve`, `lyapunov_solve`, `pca_transform`, and
`pca_inverse_transform` all accept `out=` wherever the Rust core already has `*_into` coverage.
`compute_pca(...)`, `compute_pca_complex(...)`, `linear_regression(...)`, and
`linear_regression_complex(...)` now also accept typed `out=` result buffers
(`PcaResult` / `RegressionResult`) under the existing public names, so repeated ML workflows do
not have to allocate fresh Python result arrays on every call.
`svd_pseudo_inverse(...)` can also consume a previously computed `SvdResult` directly, so repeated
pseudo-inverse workflows can reuse decomposition factors instead of recomputing SVD from the
original matrix.
`polar_compute(...)` and `matrix_log_svd(...)` now follow that same SVD-derived reuse story:
`polar_compute(...)` can consume a typed `SvdResult` with optional typed `out=PolarResult(...)`
reuse, and `matrix_log_svd(...)` can consume `SvdResult` directly instead of recomputing the
decomposition.
The real symmetric eigen-backed matrix-function helpers now follow the same pattern too:
`matrix_exp_eigen(...)`, `matrix_log_eigen(...)`, `matrix_power(...)`, and `matrix_sign(...)` can
consume a typed `EigenResult` directly with optional `out=` reuse, while `workspace=` remains a
matrix-input-only contract on those factor-backed calls. `qr_solve_least_squares(...)` now also
accepts both direct matrix `out=` reuse and typed `QrResult` reuse for square/tall factorizations,
and `svd_null_space(...)` can reuse `SvdResult` when it retains a full right-singular basis
(`vt` square).
Tensor reconstruction/projection/contraction helpers now follow that same contract where the Rust
core already exposes a truthful `*_into` path: `tensor_hosvd_nd_reconstruct`,
`tensor_hosvd3_reconstruct`, `tensor_tucker_project`, `tensor_tucker_expand`, `tensor_einsum`,
`tensor_einsum_complex`, `tensor_cp_als3_reconstruct`, `tensor_cp_als_nd_reconstruct`, and
`tensor_tt_svd_reconstruct` all accept caller-provided `out=` arrays instead of forcing fresh
tensor materialization on every call.
Dense iterative solves now follow the same pattern: `conjugate_gradient(...)`, `gmres(...)`,
`conjugate_gradient_complex(...)`, and `gmres_complex(...)` all accept `out=` under the existing
public names, and the complex iterative bindings now use the same view-first NumPy ingress
contract as the real rows instead of a separate special-case path.
Complex PCA/regression/stats rows are now also explicitly covered on Fortran-order / strided
NumPy inputs, and the remaining complex regression/statistics raw bindings now follow the same
shared helper-based view-first boundary instead of bespoke typed-array paths.
`qr_reconstruct_matrix(...)` now follows the same Rust-backed `out=` contract for both direct and
pivoted QR results, and `CholeskyResult` can now be passed back into `cholesky_solve(...)` /
`cholesky_inverse(...)` for repeated factor reuse instead of re-factorizing the original matrix.
`LuResult` now follows that same pattern for real LU workflows: `lu_solve(...)`,
`lu_inverse(...)`, `lu_determinant(...)`, and `lu_log_determinant(...)` all accept the typed
factor result directly, and the solve/inverse rows now also accept `out=` under the existing
public names.
Provider-bound mixed-precision refinement helpers are now surfaced explicitly too:
`lu_solve_mixed(...)`, `sylvester_solve_mixed(...)`, and `lyapunov_solve_mixed(...)` return typed
Python result objects carrying both the solved array and `refinement_iterations`. Those rows
require a source build with `magma-system` and intentionally admit only the truthful mixed-
provider dtypes (`float64` / `complex128`).

Repeated pairwise cosine, matrix-function, and Sylvester/Lyapunov workloads now also expose
reusable Python workspace objects (`PairwiseCosineWorkspace`, `MatrixFunctionWorkspace`,
`SylvesterWorkspace`) through the existing public APIs via `workspace=`. Schur decomposition now
follows the same explicit reuse contract: `schur_compute(...)` accepts `out=SchurResult(...)` for
caller-provided result buffers and `workspace=SchurWorkspace(...)` for repeated workloads.

Optional provider/backend/Arrow support on the Python side is a source-build workflow using the
same Cargo feature names as the Rust facade (`openblas-system`, `openblas-static`,
`netlib-system`, `netlib-static`, `magma-system`, `accelerator-rayon`, `accelerator-wgpu`,
`arrow`). Inspect the installed Python extension with `pynabled.build_features()`. For build
instructions and host/toolchain requirements, see [BUILD.md](BUILD.md).

To publish **pynabled** wheels to PyPI (tags, CI, TestPyPI), see [docs/PYPI_PUBLISH.md](docs/PYPI_PUBLISH.md).

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
6. `magma-system`: enables NVIDIA MAGMA provider-backed decomposition paths.
7. `accelerator-rayon`: enables selected parallel CPU kernels.
8. `accelerator-wgpu`: enables WGPU-backed dense/vector/tensor kernel paths (`f32` native, `f64` native when `SHADER_F64` is available).
9. `arrow`: enables facade-only Arrow/ndarray interop adapters backed by `ndarrow`.

```toml
[dependencies]
nabled = { version = "0.0.8", features = ["openblas-system"] }
```

```toml
[dependencies]
nabled = { version = "0.0.8", features = ["arrow"] }
```

Arrow interop notes:

1. Arrow awareness is isolated to facade crate `nabled`; lower crates remain ndarray-native.
2. Arrow checkpoint 2 is complete under the concept-first standalone / `rows-of-X` contract for
   dense, sparse, decomposition, tensor, batched, and ML/stat workflows.
3. Arrow wrappers delegate to the same ndarray-native execution paths, so provider backends,
   GPU/backend routing, and MAGMA behavior are inherited rather than reimplemented.
4. The bridge crate used by the facade is re-exported as `nabled::ndarrow` behind feature `arrow`.
5. Exact direct-ingress coverage is tracked in `docs/ARROW_SUPPORT_MATRIX.md`.

Feature behavior:

1. `openblas-system` implies `blas`.
2. Provider feature selection (`openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`) is compile-time and internal to decomposition paths.
3. `magma-system` implies provider-backed decomposition routing and composes with the OpenBLAS/LAPACK provider stack.
4. Backend acceleration is compile-time and kernel-family-specific.
5. GPU-backend-dispatched kernels use explicit CPU fallback when no usable GPU is available.
6. `f64` native GPU execution depends on `wgpu::Features::SHADER_F64`; when unavailable, backend-dispatched `f64` calls fall back to CPU.
7. Provider/toolchain requirements depend on provider choice; `openblas-static` and
   `netlib-static` require native build toolchains (`gcc`/`gfortran`/`make`), and `netlib-system`
   requires a system `LAPACK`/Fortran runtime available to the linker.

## Arrow Interop

`nabled` is now `ndarrow`-powered behind feature `arrow`.

Current Arrow-ingress coverage includes:

1. Canonical dense vector batches over `FixedSizeList<T>(D)`
2. Canonical sparse CSR object rows and sparse matrix batches, including `ndarrow.csr_matrix` and `ndarrow.csr_matrix_batch`, with direct sparse solve/product/reuse workflows
3. Canonical dense fixed-shape and variable-shape tensor batches
4. LU, Cholesky, QR, SVD, Eigen, Schur, Polar, matrix-functions, triangular solves
5. Batched decomposition helpers
6. Iterative solvers, Jacobian tools, optimization, PCA, regression, and stats
7. Real and complex workflows where the Arrow boundary contract is explicit and natural

For exact module-by-module coverage and intentional remaining gaps, see:

1. `docs/NDARROW_INTEGRATION.md`
2. `docs/ARROW_SUPPORT_MATRIX.md`

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
