# Nabled Production-Ready Plan

## Purpose

This document defines the path from the current pre-release state to a production-ready linear algebra library that can be used confidently in place of existing options.

Production-ready here means:

1. Reliable numerical behavior with explicit error semantics.
2. Competitive or better performance on representative workloads.
3. Clear API stability and maintenance guarantees.
4. Strong test and benchmark process with enforced quality gates.

## Current Scope Snapshot

Nabled currently provides:

1. Dense decompositions: SVD, QR, LU, Cholesky, Schur, Polar, Eigen.
2. Solvers: least squares, triangular solve, Sylvester/Lyapunov, CG/GMRES.
3. Higher-level methods: PCA, linear regression.
4. Numerical differentiation: Jacobian, gradient, Hessian (real + complex).
5. Matrix functions: exp/log/power/sign.
6. Statistics: means, centering, covariance, correlation.

## Competitive Landscape (What To Compare Against)

The closest crates to current functionality:

1. `nalgebra`
2. `nalgebra-lapack`
3. `ndarray-linalg`
4. `faer`
5. `finitediff` (for differentiation benchmark coverage)

Benchmark intent:

1. Define where nabled should be fastest.
2. Define where nabled should differentiate if not fastest.

## Differentiation Strategy

Primary positioning:

1. Dual dense backend support with consistent module-level API concepts.
2. Combined coverage of decomposition + solver + matrix functions + numerical differentiation.
3. Strong correctness contracts and high-confidence validation discipline.

Secondary positioning:

1. Best-in-class "performance plus ergonomics" for dense scientific and ML workflows in Rust.
2. Predictable error handling and no hidden panic behavior in public APIs.

## Gap Analysis (Production-Grade Expectations)

Major gaps to close:

1. Backend/performance architecture for high-performance engines and low-copy paths.
2. Broader benchmark suite across modules and competitors, with reproducible reporting.
3. Sparse/structured matrix roadmap and compatibility plan.
4. Complex-number consistency across all modules.
5. API stability policy and release process (semver, deprecations, migration notes).
6. Numerical-contract documentation (residual expectations, tolerance semantics, failure modes).

## Execution Plan

## Phase 0: Benchmark and Quality Baseline (Start Now)

This is established immediately and remains a permanent gate.

Deliverables:

1. Benchmark matrix covering decompositions, solves, iterative methods, PCA/regression, differentiation.
2. Competitor benchmark harness for `nalgebra`, `faer`, `nalgebra-lapack`, `ndarray-linalg`, `finitediff` where applicable.
3. Accuracy checks tied to benchmark runs (speed without correctness does not count).
4. CI gates for:
   - test pass
   - clippy pass
   - coverage threshold
   - benchmark regression checks (at least smoke-level in CI, full suite on scheduled runs)

## Phase 1: Backend/Performance Architecture (Top Priority)

Goal:

1. Make adding backends straightforward.
2. Keep runtime overhead at zero (or near-zero where truly unavoidable).

Design principles:

1. Static dispatch first. No trait objects in hot paths.
2. Domain-specific traits instead of one giant abstraction trait.
3. Backend-specific kernels avoid cross-backend conversion in critical routines.
4. Feature-gated backend compilation to keep binary size and compile time controlled.
5. Explicit capability modeling (for example: full SVD available, generalized eigen unavailable) at compile time.

Proposed architecture:

1. `src/backend/` internal layer:
   - `svd.rs`, `qr.rs`, `lu.rs`, `eigen.rs`, `schur.rs`, `iterative.rs`, etc.
   - traits such as `SvdKernel`, `QrKernel`, `LuKernel`, each with minimal required methods.
2. `src/backend/nalgebra_impl/` and `src/backend/ndarray_impl/`:
   - direct kernel implementations for current backends.
3. Optional new implementations:
   - `src/backend/faer_impl/`
   - `src/backend/lapack_impl/`
4. Public API remains ergonomic:
   - existing module style can remain for compatibility
   - internally routed through kernel layer to remove duplication.

Zero-overhead constraints:

1. Generic monomorphized kernels (`#[inline]` where appropriate).
2. No dynamic dispatch in core numeric loops.
3. No heap allocations introduced solely by abstraction layers.
4. No hidden matrix format conversion on performance-critical calls.
5. Benchmarks must prove no regression versus current direct implementations.

Migration strategy:

1. Start with one domain (`svd` + `qr`) as architecture pilot.
2. Validate parity and performance.
3. Expand module by module after pilot passes.

### Domain Trait Scope (No Patchwork Rule)

To avoid fragmented backend support, backend abstraction is rolled out by domain tiers.

Tier A: Core kernel domains (required for a backend to be considered supported)

1. `SvdKernel`
2. `QrKernel`
3. `LuKernel`
4. `CholeskyKernel`
5. `EigenKernel`
6. `SchurKernel`
7. `TriangularSolveKernel`

Tier B: Derived domains (may compose Tier A kernels)

1. Polar decomposition
2. PCA
3. Regression
4. Matrix functions
5. Sylvester/Lyapunov

Tier C: Utility and extension domains

1. Iterative solvers
2. Statistics helpers
3. Numerical differentiation

Support levels:

1. **Experimental backend**: Pilot-only (`svd` + `qr`) to validate architecture. Not advertised as full backend support.
2. **Supported backend**: Full Tier A parity with tests and benchmarks.
3. **Production backend**: Tier A + Tier B parity, documented caveats, and stable benchmark profile.

Backend acceptance rule:

1. Do not mark a backend as supported if Tier A is partial.
2. Keep a backend capability matrix in docs and CI checks to prevent regressions.

## Phase 2: Backend Expansion

Recommended order:

1. Keep current `nalgebra` and `ndarray` paths as baseline.
2. Add one high-performance backend first (`faer` or LAPACK-backed path).
3. Add provider-specific LAPACK support for head-to-head performance targets.

Acceptance for each backend:

1. Functional parity matrix defined per module.
2. Error and edge-case behavior parity tested.
3. Benchmark report generated and reviewed.

## Phase 3: Full-Feature Replacement Readiness

After backend architecture is stable:

1. Expand sparse/structured roadmap (or publish explicit non-goal for initial release).
2. Broaden complex support where missing.
3. Finalize API stabilization policy and release process.
4. Publish production documentation and migration guidance.

## Benchmark Program (Permanent)

Minimum benchmark dimensions:

1. Sizes: small, medium, large.
2. Shapes: square, tall-skinny, wide-short.
3. Types: `f32`, `f64`.
4. Workloads:
   - decomposition-only
   - solve workflows
   - end-to-end tasks (for example PCA fit+transform, regression fit+predict)
5. Correctness metrics:
   - residual norms
   - reconstruction error
   - tolerance pass/fail.

## Unsafe Policy (Performance vs Safety)

Short answer: mostly yes, you can have both, with discipline.

Policy:

1. Safe-by-default in public and orchestration layers.
2. `unsafe` allowed only in tightly scoped backend internals when there is measured performance benefit or required FFI interop.
3. Every unsafe block requires:
   - documented invariants
   - boundary tests
   - debug assertions for preconditions where possible
   - coverage by sanitizer/miri/fuzz or equivalent validation.
4. No "blanket no-unsafe" pledge if it prevents justified performance and interop goals.
5. No unnecessary unsafe for micro-optimizations without benchmark evidence.

Practical interpretation:

1. First exhaust safe optimizations (algorithm choice, allocation reduction, memory locality, inlining, specialization).
2. Introduce unsafe only for clear hotspots or interop boundaries.
3. Keep unsafe localized so the rest of the system remains easy to reason about.

## Production Gate Checklist

1. Coverage and tests above target thresholds.
2. Benchmark suite broad and reproducible.
3. Backends architecture in place with at least one high-performance backend beyond baseline.
4. Documented numerical contracts and error semantics.
5. API stability/release policy documented.
6. CI enforces all of the above continuously.

## Immediate Next Actions

1. Create benchmark spec document and result format in `docs/`.
2. Implement backend kernel pilot for `svd` and `qr`.
3. Add one external competitor benchmark per pilot domain.
4. Review pilot results before scaling architecture across all modules.

### Status Note (February 27, 2026)

Current state:

1. Benchmark specification is defined in `docs/BENCHMARK_SPEC.md`.
2. Backend kernel pilot is implemented for `svd` and `qr` using internal `SvdKernel` and `QrKernel` adapters.
3. Public APIs remain unchanged; dispatch routes through backend-owned kernel implementations (no temporary round-trip back into domain `*_impl` functions).
4. Competitor benchmark harnesses now exist for both pilot domains (`benches/svd_benchmarks.rs`, `benches/qr_benchmarks.rs`) with direct `nalgebra` comparison paths and correctness guards.
5. Benchmark artifact reporting is implemented via `src/bin/benchmark_report.rs` and `just bench-report`, emitting `coverage/benchmarks/summary.json`, `summary.csv`, and `regressions.md`.
6. Next execution focus is adding additional competitor backends before extending kernels across the rest of Tier A.
