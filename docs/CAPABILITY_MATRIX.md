# Capability Matrix

Last updated: 2026-03-01

## Purpose

This document is the canonical scope and gap map for `nabled`.

It serves two goals:
1. Track what is implemented today.
2. Define what must exist for the target production scope.

Operational sequencing (`Done / Next / Needed`) lives in `docs/EXECUTION_TRACKER.md`.

## Status Legend

- `Implemented`: shipped in public API and covered by tests.
- `Partial`: shipped, but missing depth (API shape, perf path, numeric breadth, or edge-case coverage).
- `Missing`: not currently provided by `nabled` APIs.

## Current Capability Inventory

| Area | Capability | Package/Module | Status | Benchmark Coverage | Notes |
|---|---|---|---|---|---|
| Core validation | shape checks for matrix/system inputs | `nabled-core::validation` | Implemented | No | Shared helpers exist; error model still minimal. |
| Core errors | common shape errors + shared taxonomy (`NabledError`) | `nabled-core::errors` | Implemented | No | Domain errors remain local, but normalization path exists via `IntoNabledError`. |
| Cholesky | factorization, solve, inverse (+ complex paths) | `nabled-linalg::cholesky` | Implemented | Yes | Complex APIs execute in both internal and provider-enabled builds. Bench exists (`cholesky_benchmarks`). |
| Eigen | symmetric/generalized/non-symmetric dense eigen | `nabled-linalg::eigen` | Implemented | Yes | Non-symmetric APIs now run in internal mode (complex Schur/closed-form small cases) and provider mode (`Eig`). |
| LU | factorization, solve, inverse, det/logdet (+ complex paths) | `nabled-linalg::lu` | Implemented | Yes | Complex solve/inverse/determinant APIs execute in both internal and provider-enabled builds. Bench exists (`lu_benchmarks`). |
| QR | full/reduced QR, pivoting, least-squares | `nabled-linalg::qr` | Implemented | Yes | Bench exists (`qr_benchmarks`). |
| SVD | full/truncated/toleranced SVD, rank, cond, pinv, null space | `nabled-linalg::svd` | Implemented | Yes | Real and complex paths execute in both internal and provider-enabled builds. Bench exists (`svd_benchmarks`) with complex cases. |
| Triangular solves | lower/upper substitution (+ complex variants) | `nabled-linalg::triangular` | Implemented | Yes | Includes allocation-controlled `*_into` paths and complex solve entrypoints. |
| Vector primitives | dot/norm/cosine/pairwise/batched dot (+ complex Hermitian baseline) | `nabled-linalg::vector` | Implemented | Yes | Bench exists (`vector_benchmarks`) with ndarray competitor baselines. |
| Matrix primitives | matvec/matmat + batched matrix-kernels (`*_into`, views) | `nabled-linalg::matrix` | Implemented | No | Dense pipeline APIs exist as first-class nabled surfaces, including batched matrix-matrix kernels. |
| Schur | Schur decomposition | `nabled-linalg::schur` | Implemented | Yes | Includes complex parity in both internal and provider-enabled builds; bench exists (`schur_benchmarks`) with manual competitor baseline. |
| Polar | polar decomposition (+ complex variant) | `nabled-linalg::polar` | Implemented | Yes | Complex variant executes in both internal and provider-enabled builds; dedicated benchmark exists (`polar_benchmarks`) with complex cases. |
| Sylvester/Lyapunov | dense equation solves | `nabled-linalg::sylvester` | Implemented | Yes | Includes complex parity in both internal and provider-enabled builds; bench exists (`sylvester_benchmarks`) with manual competitor baseline. |
| Matrix functions | exp/log/power/sign | `nabled-linalg::matrix_functions` | Implemented | Yes | Includes complex `exp/log/power/sign` coverage in both internal and provider-enabled builds. Bench exists (`matrix_functions_benchmarks`) with complex cases. |
| Orthogonalization | Gram-Schmidt variants | `nabled-linalg::orthogonalization` | Implemented | Yes | Dedicated benchmark now exists (`orthogonalization_benchmarks`). |
| Iterative solvers | CG, GMRES | `nabled-ml::iterative` | Implemented | No | Good foundation for larger optimization stack. |
| Sparse kernels | CSR/CSC/COO primitives, sparse matvec/matmat, Jacobi/Gauss-Seidel/CG/PCG/BiCGSTAB, ILU(0) + ILU0-preconditioned BiCGSTAB | `nabled-linalg::sparse` | Implemented | Yes | Includes CSR↔CSC conversion, sparse-sparse multiplication, and ILU0 factorization/preconditioned solve paths. Bench exists (`sparse_benchmarks`) with dense ndarray baseline. |
| Optimization | line search, gradient descent, Adam, momentum, RMSProp | `nabled-ml::optimization` | Implemented | Yes | Bench exists (`optimization_benchmarks`) with manual baseline loops. |
| Tensor/cube primitives | batched cube kernels + higher-rank last-axis `ArrayD` ops | `nabled-linalg::tensor` | Partial | Yes | Rank-3 APIs are present with owned/view/into variants; higher-rank baseline now includes last-axis reductions/normalization/batched dot. |
| Accelerator contracts | compile-time backend trait + CPU execution/chunking + feature-gated accelerated matmat + unsupported CUDA/distributed placeholders | `nabled-linalg::accelerator` | Partial | No | Establishes compile-time backend seam; accelerated CPU path (`accelerator-rayon`) exists, concrete GPU/distributed kernels remain open. |
| Jacobian tools | numerical Jacobian/gradient/Hessian | `nabled-ml::jacobian` | Implemented | No | Finite-difference based. |
| PCA | PCA + transform/inverse-transform | `nabled-ml::pca` | Implemented | No | |
| Regression | linear regression | `nabled-ml::regression` | Implemented | No | |
| Stats | means/centering/covariance/correlation | `nabled-ml::stats` | Implemented | No | |

## Target Scope Matrix (Aligned to Project Goals)

### P0: Required for "foundation production-ready" nabled

| Capability Group | Current Status | Gap |
|---|---|---|
| Stable ndarray-first dense decomposition suite | Implemented | Dense decomposition APIs now include non-symmetric eigen coverage across internal/provider modes; remaining work is performance tuning, not missing baseline capability. |
| Vector-first primitives for embeddings workflows | Implemented | Dot/norm/cosine/pairwise distance/batched dot are available; sparse and higher-rank follow-ons remain. |
| Matrix-vector and matrix-matrix pipeline primitives | Implemented | First-class `nabled-linalg::matrix` APIs now cover matvec/matmat and batched matrix operations with owned/view/into variants. |
| Unified error taxonomy and API contracts | Implemented | Shared taxonomy + stable mapping tests now include newly added matrix/tensor/accelerator/sparse breadth paths. |
| Performance-contract APIs (explicit allocations/workspaces) | Partial | `*_into` and workspace patterns now include vector/triangular/cholesky/svd/qr/matrix_functions/schur/sylvester; copy-elision has advanced in ML view paths (`stats`/`regression`/`pca`) with remaining convenience wrappers still open. |
| Numeric robustness controls | Implemented | Dense-kernel tolerance and iteration defaults are centralized via `internal::DenseKernelPolicy`; additional domain-specific tuning can layer on top. |
| Benchmark coverage for all Tier-A kernels | Implemented | Dedicated suites now include polar/orthogonalization, with practical manual competitor baselines in schur/sylvester. |

### P1: Required for broader "go-to" linalg/ML scope

| Capability Group | Current Status | Gap |
|---|---|---|
| Batched operations over many vectors/matrices | Implemented | Vector, dense matrix, sparse matrix, and cube-level batched primitives are now exposed with explicit allocation-control paths. |
| Sparse linear algebra primitives | Implemented | Baseline sparse primitives now span CSR/CSC/COO formats, sparse-dense/sparse-sparse products, iterative solve breadth including `BiCGSTAB`, and ILU(0)-backed preconditioned solve paths. |
| Complex-number parity across major algorithms | Partial | Complex parity now includes vector kernels, QR, SVD, LU, Cholesky, Schur, Sylvester/Lyapunov, matrix-functions (`exp/log/power/sign`), polar decomposition, and triangular solves across internal/provider builds; this is now validated by facade integration tests plus complex benchmark visibility in selected dense suites. Many other domains are still f64-only. |
| Non-symmetric dense eigen coverage | Implemented | Non-symmetric real/complex eigen APIs exist with internal and provider-enabled execution paths. |
| More optimization primitives | Implemented | First-order suite now includes line search, gradient descent, Adam, momentum, and `RMSProp`; advanced second-order and constrained methods remain future enhancements. |

### P2: Out of immediate scope (documented future direction)

| Capability Group | Current Status | Gap |
|---|---|---|
| Tensor/cube-focused higher-rank APIs | Partial | Rank-3 cube primitives plus baseline last-axis `ArrayD` operations are present; broader tensor algebra/contraction depth is still missing. |
| GPU/distributed kernels | Partial | Compile-time accelerator contracts and feature-gated accelerated CPU matmat are present; concrete GPU/distributed kernels are not yet implemented. |
| Arrow-aware API surface in `nabled` | Intentionally omitted | Per project decision, Arrow interop belongs to downstream crates. |

## Sufficiency Verdict

`nabled` is sufficient as a strong ndarray-native dense-core base, but not yet sufficient for the full target scope described for embedding-centric and broad production workflows.

Concretely, the largest missing pieces are now:
1. Further copy-elision/performance-contract hardening across remaining convenience/view wrappers.
2. Sparse depth beyond the current iterative/preconditioned baseline (factorization-grade sparse workflows).
3. Concrete accelerator/tensor depth beyond the new baseline seams (actual GPU/distributed kernels and broader higher-rank tensor algebra).

## Execution Order Driven by This Matrix

1. Continue replacing allocating convenience wrappers with no-copy view-aware kernels where feasible.
2. Expand sparse depth from baseline primitives into factorization-oriented workflows.
3. Expand tensor APIs beyond current rank-3 + last-axis `ArrayD` baseline toward broader higher-rank semantics.
4. Convert accelerator contracts from CPU/feature-gated baseline into concrete GPU/distributed kernels.

## Definition of Done for This Document

When updating this matrix:
1. Keep every capability tied to an actual module/API.
2. Mark status using the legend only.
3. Update the verdict if scope coverage changes materially.

## Post-Readiness Hardening Backlog

These are intentionally tracked as post-capability polish passes once P0/P1 sufficiency is met.

1. Benchmark outlier analysis and targeted remediation for regressions or weak domains.
2. Allocation audit to ensure heap allocations occur only where contractually necessary.
3. SIMD opportunity pass for hand-rolled kernels that dominate wall-clock time.
4. Threading policy pass for internal parallelism, including `accelerator-rayon` interactions with BLAS/LAPACK provider threading to avoid oversubscription.
