# Status Snapshot

Last updated: 2026-03-01

## Summary

Workspace migration for library domains is complete.

1. Workspace members exist: `nabled-core`, `nabled-linalg`, `nabled-ml`.
2. `crates/nabled` is the facade package re-exporting workspace crates.
3. `crates/nabled/src/` contains facade/library entrypoint and binary tooling only.
4. Backend/feature model now uses `blas` + provider features (`openblas-system` first).
5. Public `*_lapack` compatibility wrappers have been removed.
6. Dense-kernel APIs are normalized around `decompose`/domain-specific operation naming.
7. Vector primitives are available in `nabled-linalg::vector` with pairwise and batched APIs.
8. Explicit allocation paths (`*_into`) and reusable workspace pattern are in place for key hot paths.
9. Tier-A benchmark surface expanded beyond four suites (LU/Cholesky/Eigen/Vector added).
10. Sparse baseline is now present (`CSR`, sparse matvec, Jacobi sparse solve).
11. Sparse baseline expanded with `COO` + COO→CSR conversion and Gauss-Seidel solve.
12. New benchmark suites now cover `sparse`, `schur`, `sylvester`, `optimization`, `polar`, and `orthogonalization`.
13. Complex decomposition parity started (`QR` complex path and complex SVD).
14. Shared cross-domain taxonomy exists via `nabled-core::errors::NabledError`.
15. Complex-number vector parity is now present (Hermitian dot, complex norm/cosine).
16. First-order optimization primitives are now present (line search, gradient descent, Adam).
17. View-first (`ArrayView*`) API coverage is expanded across remaining heavy linalg/ml domains with parity tests.
18. Public API namespaces are flattened; `ndarray_*` wrapper modules are removed in favor of direct domain APIs.
19. Complex parity now extends beyond QR/SVD into LU, Cholesky, Schur, Sylvester/Lyapunov, matrix-functions (`exp/log/power/sign`), polar decomposition, and triangular solves; these domains now execute in both internal and provider-enabled builds.
20. Dedicated `IntoNabledError` mapping-stability tests now exist in `nabled-linalg` and `nabled-ml`.
21. Benchmark competitor baselines now exist for vector, sparse, optimization, schur, and sylvester suites.
22. Targeted per-file coverage lift for `eigen`, `lu`, `orthogonalization`, and `polar` is complete.
23. Allocation behavior for allocating view/convenience APIs is now explicitly documented in rustdoc.
24. Complex parity validation matrix is now integration-anchored at facade level (parity + error mapping tests) and benchmark-visible in complex `svd`, `matrix_functions`, and `polar` smoke suites.
25. Dense numerical policy is now centralized (`internal::DenseKernelPolicy`) and applied consistently across dense kernel tolerance/iteration defaults.
26. Dense pipeline primitives are now first-class (`nabled-linalg::matrix`) including batched matrix-matrix APIs.
27. Sparse breadth now includes CSC format support, CSR↔CSC conversion, sparse-sparse multiplication, and `BiCGSTAB`.
28. Optimization breadth now includes momentum descent and `RMSProp`.
29. Non-symmetric dense eigen paths now run in both internal and provider-enabled modes.
30. Initial P2 foundations now exist: `nabled-linalg::tensor` (cube primitives) and `nabled-linalg::accelerator` (compile-time backend contracts).
31. ML copy-elision improved in hot view paths (`stats`, `regression`, `pca` transform/inverse transform) by routing through view-native internals.
32. Sparse breadth now includes Jacobi preconditioning and preconditioned conjugate gradient (`PCG`).
33. Tensor breadth now includes higher-rank `ArrayD` last-axis operations (`sum`, norm, normalize, batched dot).
34. Accelerator domain now includes explicit serial matmat and feature-gated accelerated matmat (`accelerator-rayon`) with strict error mapping.
35. Benchmark/report coverage now includes new matrix/tensor suites and expanded sparse cases, with classifier support in reporting tooling.
36. Sparse depth now includes ILU(0) factorization and ILU0-preconditioned `BiCGSTAB` for non-symmetric systems.
37. Copy-elision has now advanced through additional core linalg domains (`orthogonalization`, `qr`, `svd`, `schur`, `sylvester`) via view-native dispatch paths.
38. Copy-elision has now advanced further through `lu`, `cholesky`, `eigen`, and `triangular` view APIs using view-native validation/dispatch in internal mode.
39. Remaining convenience view wrappers in `polar` and `pca` now use view-native internals (no wrapper-level hidden `to_owned()` allocations), and provider dispatch in `lu`/`cholesky`/`eigen` is view-native.
40. Sparse preconditioning depth now includes IC(0) (`ic0_factor`, `apply_ic0_preconditioner`) and IC(0)-preconditioned `PCG` (`pcg_ic0_solve`), with tests and benchmark coverage.

## Current Code Ownership

1. `crates/nabled-core`
   - shared prelude, validation, and core error scaffolding.
2. `crates/nabled-linalg`
   - decomposition, solver, and matrix-function domains:
     `svd`, `qr`, `lu`, `cholesky`, `eigen`, `schur`, `polar`, `sylvester`,
     `matrix_functions`, `orthogonalization`, `triangular`, `vector`, `matrix`,
     `sparse`, `tensor`, `accelerator`.
3. `crates/nabled-ml`
   - ML/statistics-oriented domains:
     `iterative`, `jacobian`, `pca`, `regression`, `stats`.
4. `crates/nabled/src/` (facade crate)
   - facade `lib.rs` and binary/reporting tools only.

## Constraints In Force

1. ndarray-first API model.
2. No nalgebra dependencies or code paths.
3. No hidden conversion-heavy hot paths.
4. Quality gates remain strict (`just checks`, clippy `-D warnings`, tests, coverage >= 90%).
5. Backend selection is compile-time only; no runtime backend dispatch.
6. Public APIs should remain backend-agnostic.

## Operational Notes

1. On macOS, provider-enabled `just` recipes now inject Homebrew OpenBLAS env (`PKG_CONFIG_PATH`, `OPENBLAS_DIR`) automatically.
2. This avoids per-shell setup drift for common local quality/bench workflows.
3. Quality gates run both internal mode and provider-enabled mode in local `just checks` and CI.
4. Coverage is scoped to library surfaces (`--lib --tests`) with non-library exclusions (`errors|examples|benches|src/bin`) and now hard-fails below `90%` lines.

## Next Required Milestone

Harden workspace contracts and release readiness:

1. Continue copy-elision pass for remaining convenience/view wrappers that still allocate hidden owned buffers.
2. Expand sparse depth from current ILU(0) baseline into broader factorization-grade sparse workflows.
3. Expand tensor beyond current last-axis `ArrayD` baseline and advance accelerator from CPU parallel baseline toward concrete GPU/distributed kernels.
4. Keep execution updates current in `docs/EXECUTION_TRACKER.md`.

## Completion Criteria For Migration

1. Facade package is `crates/nabled` and root manifest is virtual-workspace-only.
2. Domain modules live in `crates/nabled-linalg` and `crates/nabled-ml`.
3. CI, benches, examples, and tests run workspace-wide without root-implementation coupling.
