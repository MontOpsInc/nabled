# Capability Matrix

Last updated: 2026-03-03

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
| Eigen | symmetric/generalized/non-symmetric dense eigen (+ balancing and left/right non-symmetric surfaces) | `nabled-linalg::eigen` | Implemented | Yes | Non-symmetric APIs run in internal/provider modes; balancing (`balance_nonsymmetric*`) and matched left/right surfaces (`nonsymmetric_bi*`) are now available. |
| LU | factorization, solve, inverse, det/logdet (+ complex paths) | `nabled-linalg::lu` | Implemented | Yes | Complex solve/inverse/determinant APIs execute in both internal and provider-enabled builds. Bench exists (`lu_benchmarks`). |
| QR | full/reduced QR, pivoting, least-squares | `nabled-linalg::qr` | Implemented | Yes | Includes true column-pivoted QR (`P` is no longer identity-placeholder) and least-squares support for both overdetermined and underdetermined (minimum-norm) systems. Bench exists (`qr_benchmarks`). |
| SVD | full/truncated/toleranced SVD, rank, cond, pinv, null space | `nabled-linalg::svd` | Implemented | Yes | Real and complex paths execute in both internal and provider-enabled builds. Bench exists (`svd_benchmarks`) with complex cases. |
| Triangular solves | lower/upper substitution (+ complex variants) | `nabled-linalg::triangular` | Implemented | Yes | Includes allocation-controlled `*_into` paths and complex solve entrypoints. |
| Vector primitives | dot/norm/cosine/pairwise/batched dot (+ complex Hermitian baseline) | `nabled-linalg::vector` | Implemented | Yes | Stable allocating `dot`/pairwise entrypoints are compile-time backend-kernel dispatched; bench exists (`vector_benchmarks`) with ndarray competitor baselines. |
| Matrix primitives | matvec/matmat + batched matrix-kernels (`*_into`, views, broadcasted batch matmat) | `nabled-linalg::matrix` | Implemented | No | Dense pipeline APIs exist as first-class nabled surfaces, including real/complex parity and batched/broadcast semantics; stable allocating owned entrypoints (`matvec`, `matmat`, `batched_matmat`, `batched_row_matvec`) dispatch through compile-time backend kernels. |
| Batched decomposition helpers | batched QR/SVD/LU/Cholesky/symmetric eigen over matrix stacks | `nabled-linalg::batched` | Implemented | No | Batch entrypoints expose decomposition-level workflows without requiring caller-side loops. |
| Schur | Schur decomposition | `nabled-linalg::schur` | Implemented | Yes | Includes complex parity in both internal and provider-enabled builds; bench exists (`schur_benchmarks`) with manual competitor baseline. |
| Polar | polar decomposition (+ complex variant) | `nabled-linalg::polar` | Implemented | Yes | Complex variant executes in both internal and provider-enabled builds; dedicated benchmark exists (`polar_benchmarks`) with complex cases. |
| Sylvester/Lyapunov | dense equation solves | `nabled-linalg::sylvester` | Implemented | Yes | Includes complex parity in both internal and provider-enabled builds; bench exists (`sylvester_benchmarks`) with manual competitor baseline. |
| Matrix functions | exp/log/power/sign | `nabled-linalg::matrix_functions` | Implemented | Yes | Includes complex `exp/log/power/sign` coverage in both internal and provider-enabled builds. Bench exists (`matrix_functions_benchmarks`) with complex cases. |
| Orthogonalization | Gram-Schmidt variants | `nabled-linalg::orthogonalization` | Implemented | Yes | Includes complex Gram-Schmidt parity and dedicated benchmark coverage (`orthogonalization_benchmarks`). |
| Iterative solvers | CG, GMRES | `nabled-ml::iterative` | Implemented | No | Includes real and complex CG/GMRES APIs. |
| Sparse kernels | CSR/CSC/COO primitives, sparse matvec/matmat, Jacobi/Gauss-Seidel/CG/PCG/BiCGSTAB/GMRES, ILU(0)/ILU(k)/IC(0)/ILUT/ILDL(0) preconditioning workflows + direct sparse LU solve/reuse paths | `nabled-linalg::sparse` | Implemented | Yes | Includes CSR↔CSC conversion, sparse-sparse multiplication, factorization-reuse solve APIs, ILU0/ILUK/ILUT/ILDL0-preconditioned GMRES/BiCGSTAB paths, and direct sparse LU solve/reuse workflows; stable allocating sparse matvec/matmat entrypoints now use compile-time backend kernel dispatch. Bench exists (`sparse_benchmarks`) with dense ndarray baseline. |
| Optimization | line search, gradient descent, Adam, momentum, RMSProp, projected GD, stochastic GD, BFGS | `nabled-ml::optimization` | Implemented | Yes | Bench exists (`optimization_benchmarks`) with manual baseline loops. |
| Tensor/cube primitives | batched cube kernels + higher-rank `ArrayD` ops (last-axis reductions, axis permutation, explicit-axis contraction, N-D batched last-two matmul) + rank-3 HOSVD + einsum-style binary contractions | `nabled-linalg::tensor` | Implemented | Yes | Required v1 tensor surface is explicit and complete across allocating/view/into forms for required operation families (real/complex where applicable); see `docs/V1_STABILITY.md` for scope lock and contracts. |
| Accelerator contracts | compile-time backend trait + per-operation kernel trait dispatch + CPU execution/chunking + feature-gated accelerated matmat + feature-gated GPU kernels (`wgpu`) + explicit CPU fallback behavior for out-of-scope v1 GPU kernels | `nabled-linalg::accelerator` | Implemented | Yes | Required v1 kernel families are wired through compile-time backend dispatch over `CpuBackend`/`GpuBackend`, with bounded GPU `f32` support and explicit fallback contracts outside v1 GPU scope; see `docs/V1_STABILITY.md` for the locked support matrix. |
| Jacobian tools | numerical Jacobian/gradient/Hessian | `nabled-ml::jacobian` | Implemented | No | Finite-difference based. |
| PCA | PCA + transform/inverse-transform | `nabled-ml::pca` | Implemented | No | |
| Regression | linear regression | `nabled-ml::regression` | Implemented | No | |
| Stats | means/centering/covariance/correlation | `nabled-ml::stats` | Implemented | No | |

## Execution Axes Model

`nabled-linalg` currently operates on three distinct execution concepts:

1. `Provider`: decomposition implementation source (`internal` or selected LAPACK provider feature).
2. `Backend`: primitive-kernel execution target (`CpuBackend`, `GpuBackend`).
3. `Kernel`: operation-family contract implemented per backend (dense/sparse/vector/tensor/triangular families; see `docs/KERNEL_CATALOG.md`).

These axes are intentionally orthogonal:

1. Provider selection controls decomposition-style paths.
2. Backend/kernel selection controls operation-kernel paths.
3. A single public algorithm may use both axes in sequence.

Canonical kernel-family scope and wiring status are tracked in `docs/KERNEL_CATALOG.md`.

## Target Scope Matrix (Aligned to Project Goals)

### P0: Required for "foundation production-ready" nabled

| Capability Group | Current Status | Gap |
|---|---|---|
| Stable ndarray-first dense decomposition suite | Implemented | Dense decomposition APIs now include non-symmetric eigen coverage across internal/provider modes; remaining work is performance tuning, not missing baseline capability. |
| Vector-first primitives for embeddings workflows | Implemented | Dot/norm/cosine/pairwise distance/batched dot are available; sparse and higher-rank follow-ons remain. |
| Matrix-vector and matrix-matrix pipeline primitives | Implemented | First-class `nabled-linalg::matrix` APIs now cover matvec/matmat and batched matrix operations with owned/view/into variants. |
| Unified error taxonomy and API contracts | Implemented | Shared taxonomy + stable mapping tests now include newly added matrix/tensor/accelerator/sparse breadth paths. |
| Performance-contract APIs (explicit allocations/workspaces) | Implemented | `*_into` and workspace patterns are established across major kernels; copy-elision audit for remaining algorithm-internal materializations is complete, with unavoidable allocations documented in `docs/PERFORMANCE_CONTRACTS.md`. |
| Numeric robustness controls | Implemented | Dense-kernel tolerance and iteration defaults are centralized via `internal::DenseKernelPolicy`; additional domain-specific tuning can layer on top. |
| Benchmark coverage for all Tier-A kernels | Implemented | Dedicated suites now include polar/orthogonalization, with practical manual competitor baselines in schur/sylvester. |

### P1: Required for broader "go-to" linalg/ML scope

| Capability Group | Current Status | Gap |
|---|---|---|
| Batched operations over many vectors/matrices | Implemented | Vector, dense matrix, sparse matrix, and cube-level batched primitives are exposed with explicit allocation-control paths; decomposition-level batch APIs now exist in `nabled-linalg::batched`. |
| Sparse linear algebra primitives | Implemented | Sparse primitives now span CSR/CSC/COO formats, sparse-dense/sparse-sparse products, iterative solve breadth including `BiCGSTAB`/`GMRES`, and ILU(0)/ILU(k)/IC(0)/ILUT/ILDL(0)-backed preconditioned solve and reuse paths. |
| Complex-number parity across major algorithms | Implemented | Complex parity now covers core linalg kernels (vector/matrix/tensor), dense decompositions/solvers (QR/SVD/LU/Cholesky/Schur/Sylvester/Triangular/Polar), matrix-functions (`exp/log/power/sign`), and iterative solvers (CG/GMRES) across internal/provider builds, with integration tests and benchmark visibility in selected dense suites. |
| Complex parity in higher-level ML/stat domains (`B-P1-006`) | Implemented | `nabled-ml` high-level domains (`stats`, `regression`, `pca`, `optimization`) now expose complex owned/view APIs with parity/error tests in quality gates. |
| Non-symmetric dense eigen coverage | Implemented | Non-symmetric real/complex APIs exist with internal/provider execution, plus balancing and matched left/right eigenvector surfaces. |
| More optimization primitives | Implemented | Optimization breadth now includes constrained (`projected_gradient_descent_box`), stochastic (`stochastic_gradient_descent`), and quasi-Newton (`bfgs`) paths in addition to first-order baselines. |

### P2: Post-v1 Expansion (documented future direction)

| Capability Group | Current Status | Gap |
|---|---|---|
| Tensor/cube-focused higher-rank APIs | Implemented | Required v1 tensor surface is complete; future expansion targets broader tensor algebra depth (for example higher-order decompositions/networks). |
| GPU and future multi-node kernels | Implemented | Required v1 bounded GPU surface is complete; future expansion targets deeper GPU dtype/op coverage and explicit multi-node orchestration. |
| Arrow-aware API surface in `nabled` | Intentionally omitted | Per project decision, Arrow interop belongs to downstream crates. |

## Sufficiency Verdict

`nabled` now satisfies the declared v1 capability scope in this matrix.

Primary remaining work:
1. Benchmark-driven optimization and outlier remediation.
2. `K-*` cleanup/normalization passes (provider/backend/kernel ownership boundaries and remaining orchestration cleanup).
3. Advanced tensor algebra depth beyond the v1 contract.
4. Advanced GPU and multi-node breadth beyond the bounded v1 GPU contract.

## Execution Order Driven by This Matrix

1. Run benchmark-driven optimization passes over hot kernels and regression outliers.
2. Execute `K-*` cleanup passes for ownership-boundary and dispatch normalization.
3. Expand tensor algebra depth beyond the locked v1 surface.
4. Expand GPU and multi-node breadth beyond the locked v1 surface.

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
