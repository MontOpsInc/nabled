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
| Cholesky | factorization, solve, inverse | `nabled-linalg::cholesky` | Implemented | Yes | Bench exists (`cholesky_benchmarks`). |
| Eigen | symmetric + generalized SPD-B eigen | `nabled-linalg::eigen` | Partial | Yes | No general non-symmetric dense eigensolver API. |
| LU | factorization, solve, inverse, det/logdet | `nabled-linalg::lu` | Implemented | Yes | Bench exists (`lu_benchmarks`). |
| QR | full/reduced QR, pivoting, least-squares | `nabled-linalg::qr` | Implemented | Yes | Bench exists (`qr_benchmarks`). |
| SVD | full/truncated/toleranced SVD, rank, cond, pinv, null space | `nabled-linalg::svd` | Implemented | Yes | Bench exists (`svd_benchmarks`). |
| Triangular solves | lower/upper substitution (+ complex variants) | `nabled-linalg::triangular` | Implemented | Yes | Includes allocation-controlled `*_into` paths and complex solve entrypoints. |
| Vector primitives | dot/norm/cosine/pairwise/batched dot (+ complex Hermitian baseline) | `nabled-linalg::vector` | Implemented | Yes | Bench exists (`vector_benchmarks`) with ndarray competitor baselines. |
| Schur | Schur decomposition | `nabled-linalg::schur` | Implemented | Yes | Bench exists (`schur_benchmarks`). |
| Polar | polar decomposition (+ complex variant) | `nabled-linalg::polar` | Implemented | No | Complex path exists; dedicated benchmark still missing. |
| Sylvester/Lyapunov | dense equation solves | `nabled-linalg::sylvester` | Implemented | Yes | Bench exists (`sylvester_benchmarks`). |
| Matrix functions | exp/log/power/sign | `nabled-linalg::matrix_functions` | Implemented | Yes | Bench exists (`matrix_functions_benchmarks`). |
| Orthogonalization | Gram-Schmidt variants | `nabled-linalg::orthogonalization` | Implemented | No | No dedicated benchmark yet. |
| Iterative solvers | CG, GMRES | `nabled-ml::iterative` | Implemented | No | Good foundation for larger optimization stack. |
| Sparse kernels | CSR/COO primitives, sparse matvec, Jacobi + Gauss-Seidel | `nabled-linalg::sparse` | Implemented | Yes | Bench exists (`sparse_benchmarks`) with dense ndarray baseline. |
| Optimization | line search, gradient descent, Adam | `nabled-ml::optimization` | Implemented | Yes | Bench exists (`optimization_benchmarks`) with manual baseline loops. |
| Jacobian tools | numerical Jacobian/gradient/Hessian | `nabled-ml::jacobian` | Implemented | No | Finite-difference based. |
| PCA | PCA + transform/inverse-transform | `nabled-ml::pca` | Implemented | No | |
| Regression | linear regression | `nabled-ml::regression` | Implemented | No | |
| Stats | means/centering/covariance/correlation | `nabled-ml::stats` | Implemented | No | |

## Target Scope Matrix (Aligned to Project Goals)

### P0: Required for "foundation production-ready" nabled

| Capability Group | Current Status | Gap |
|---|---|---|
| Stable ndarray-first dense decomposition suite | Partial | Hot dense kernels are provider-enabled; remaining depth is broader non-symmetric eigensolver coverage and provider-native depth in selected secondary domains. |
| Vector-first primitives for embeddings workflows | Implemented | Dot/norm/cosine/pairwise distance/batched dot are available; sparse and higher-rank follow-ons remain. |
| Matrix-vector and matrix-matrix pipeline primitives | Partial | Relying on ndarray directly today; nabled-level APIs/compositional helpers are missing. |
| Unified error taxonomy and API contracts | Partial | Shared taxonomy exists and is now covered by dedicated mapping-stability tests; full cross-crate error ergonomics can still be expanded. |
| Performance-contract APIs (explicit allocations/workspaces) | Partial | `*_into` and workspace patterns now include vector/triangular/cholesky/svd/qr/matrix_functions/schur/sylvester; view-first (`ArrayView*`) pass is underway. |
| Numeric robustness controls | Partial | Tolerances exist in places; policy and consistency are not unified across domains. |
| Benchmark coverage for all Tier-A kernels | Partial | Suite coverage is broad; remaining gap is dedicated benches for polar/orthogonalization and broader competitor parity in schur/sylvester domains. |

### P1: Required for broader "go-to" linalg/ML scope

| Capability Group | Current Status | Gap |
|---|---|---|
| Batched operations over many vectors/matrices | Partial | Vector batched dot and pairwise kernels exist; matrix-batch and sparse-batch APIs still missing. |
| Sparse linear algebra primitives | Partial | CSR baseline exists (`matvec`, `matvec_into`, Jacobi solve); more formats and solver breadth still missing. |
| Complex-number parity across major algorithms | Partial | Complex parity now includes vector kernels, QR, SVD, polar decomposition, and triangular solves; many other domains remain f64-only. |
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
2. Complex-number parity expansion beyond the currently covered domains (vector/QR/SVD/polar/triangular).
3. Explicit performance-contract APIs across the remaining heavy-kernel surface.
4. Broader benchmark/perf coverage for currently unbenchmarked domains (for example polar and orthogonalization) and more competitor parity for schur/sylvester.

## Execution Order Driven by This Matrix

1. Add dedicated polar and orthogonalization benchmark suites.
2. Expand competitor baselines where practical (especially schur/sylvester).
3. Extend complex-number parity into additional decomposition/solver domains.
4. Expand sparse formats and sparse solver breadth.
5. Harden cross-domain API ergonomics around the shared `NabledError` taxonomy.

## Definition of Done for This Document

When updating this matrix:
1. Keep every capability tied to an actual module/API.
2. Mark status using the legend only.
3. Update the verdict if scope coverage changes materially.
