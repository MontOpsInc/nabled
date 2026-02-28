# Capability Matrix

Last updated: 2026-02-28

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
| Core errors | common shape errors | `nabled-core::errors` | Implemented | No | Domain errors are still distributed per module. |
| Cholesky | factorization, solve, inverse | `nabled-linalg::cholesky::ndarray_cholesky` | Implemented | Yes | Bench exists (`cholesky_benchmarks`). |
| Eigen | symmetric + generalized SPD-B eigen | `nabled-linalg::eigen::ndarray_eigen` | Partial | Yes | No general non-symmetric dense eigensolver API. |
| LU | factorization, solve, inverse, det/logdet | `nabled-linalg::lu::ndarray_lu` | Implemented | Yes | Bench exists (`lu_benchmarks`). |
| QR | full/reduced QR, pivoting, least-squares | `nabled-linalg::qr::ndarray_qr` | Implemented | Yes | Bench exists (`qr_benchmarks`). |
| SVD | full/truncated/toleranced SVD, rank, cond, pinv, null space | `nabled-linalg::svd::ndarray_svd` | Implemented | Yes | Bench exists (`svd_benchmarks`). |
| Triangular solves | lower/upper substitution | `nabled-linalg::triangular` / `ndarray_triangular` | Implemented | Yes | Includes allocation-controlled `*_into` paths. |
| Vector primitives | dot/norm/cosine/pairwise/batched dot (+ complex vector baseline) | `nabled-linalg::vector::ndarray_vector` | Implemented | Yes | Bench exists (`vector_benchmarks`); complex Hermitian dot/norm/cosine landed. |
| Schur | Schur decomposition | `nabled-linalg::schur::ndarray_schur` | Implemented | No | No dedicated benchmark yet. |
| Polar | polar decomposition | `nabled-linalg::polar::ndarray_polar` | Implemented | No | No dedicated benchmark yet. |
| Sylvester/Lyapunov | dense equation solves | `nabled-linalg::sylvester::ndarray_sylvester` | Implemented | No | No dedicated benchmark yet. |
| Matrix functions | exp/log/power/sign | `nabled-linalg::matrix_functions::ndarray_matrix_functions` | Implemented | Yes | Bench exists (`matrix_functions_benchmarks`). |
| Orthogonalization | Gram-Schmidt variants | `nabled-linalg::orthogonalization::ndarray_orthogonalization` | Implemented | No | No dedicated benchmark yet. |
| Iterative solvers | CG, GMRES | `nabled-ml::iterative::ndarray_iterative` | Implemented | No | Good foundation for larger optimization stack. |
| Optimization | line search, gradient descent, Adam | `nabled-ml::optimization::ndarray_optimization` | Implemented | No | Baseline first-order primitives landed; more optimizers still possible. |
| Jacobian tools | numerical Jacobian/gradient/Hessian | `nabled-ml::jacobian::ndarray_jacobian` | Implemented | No | Finite-difference based. |
| PCA | PCA + transform/inverse-transform | `nabled-ml::pca::ndarray_pca` | Implemented | No | |
| Regression | linear regression | `nabled-ml::regression::ndarray_regression` | Implemented | No | |
| Stats | means/centering/covariance/correlation | `nabled-ml::stats::ndarray_stats` | Implemented | No | |

## Target Scope Matrix (Aligned to Project Goals)

### P0: Required for "foundation production-ready" nabled

| Capability Group | Current Status | Gap |
|---|---|---|
| Stable ndarray-first dense decomposition suite | Partial | API contract is normalized; provider-native kernels are still pending for several domains. |
| Vector-first primitives for embeddings workflows | Implemented | Dot/norm/cosine/pairwise distance/batched dot are available; sparse and higher-rank follow-ons remain. |
| Matrix-vector and matrix-matrix pipeline primitives | Partial | Relying on ndarray directly today; nabled-level APIs/compositional helpers are missing. |
| Unified error taxonomy and API contracts | Partial | Domain-local errors exist; core shared error architecture is not consolidated yet. |
| Performance-contract APIs (explicit allocations/workspaces) | Partial | `*_into` and workspace patterns now include vector/triangular/cholesky/svd/qr/matrix_functions/schur/sylvester; remaining domains still need pass. |
| Numeric robustness controls | Partial | Tolerances exist in places; policy and consistency are not unified across domains. |
| Benchmark coverage for all Tier-A kernels | Partial | Coverage expanded with LU/Cholesky/Eigen/Vector; Schur/Polar/Sylvester/ML domains still need dedicated suites. |

### P1: Required for broader "go-to" linalg/ML scope

| Capability Group | Current Status | Gap |
|---|---|---|
| Batched operations over many vectors/matrices | Partial | Vector batched dot and pairwise kernels exist; matrix-batch and sparse-batch APIs still missing. |
| Sparse linear algebra primitives | Partial | CSR baseline exists (`matvec`, `matvec_into`, Jacobi solve); more formats and solver breadth still missing. |
| Complex-number parity across major algorithms | Partial | Complex vector baseline exists; decomposition/kernel parity is still mostly f64-only. |
| Non-symmetric dense eigen coverage | Partial | Symmetric/generalized-SPD available; broader eigen support missing. |
| More optimization primitives | Partial | CG/GMRES + line search + gradient descent + Adam exist; constrained/stochastic/advanced second-order breadth is still missing. |

### P2: Out of immediate scope (documented future direction)

| Capability Group | Current Status | Gap |
|---|---|---|
| Tensor/cube-focused higher-rank APIs | Missing | `ArrayD` policy exists, APIs not yet built. |
| GPU/distributed kernels | Missing | No backend abstraction for accelerators yet. |
| Arrow-aware API surface in `nabled` | Intentionally omitted | Per project decision, Arrow interop belongs to downstream crates. |

## Sufficiency Verdict

`nabled` is sufficient as a strong ndarray-native dense-core base, but not yet sufficient for the full target scope described for embedding-centric and broad production workflows.

Concretely, the largest missing pieces are:
1. Sparse breadth beyond CSR/Jacobi and broader batch semantics beyond the current vector baseline.
2. Complex-number parity expansion beyond vector kernels.
3. Explicit performance-contract APIs across the remaining heavy-kernel surface.
4. Broader benchmark/perf coverage in Schur/Polar/Sylvester/sparse/optimization domains.

## Execution Order Driven by This Matrix

1. Add dedicated benchmarks for newly landed sparse/Schur/Sylvester/optimization domains.
2. Extend complex-number parity into decomposition kernels.
3. Expand sparse formats and sparse solver breadth.
4. Apply view-first API normalization to reduce caller-side allocations.
5. Consolidate cross-domain error taxonomy in `nabled-core`.

## Definition of Done for This Document

When updating this matrix:
1. Keep every capability tied to an actual module/API.
2. Mark status using the legend only.
3. Update the verdict if scope coverage changes materially.
