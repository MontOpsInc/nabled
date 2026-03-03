# Status Snapshot

Last updated: 2026-03-03

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
41. Sparse factorization/preconditioning now also includes ILUT (`ilut_factor`, `apply_ilut_preconditioner`) and ILUT-preconditioned `BiCGSTAB` (`bicgstab_ilut_solve`), with tests and benchmark coverage.
42. ILUT now has policy/config support (`ILUTConfig`, size-aware defaults, config-driven APIs) and a second ILUT-backed solver path via preconditioned GMRES (`gmres_ilut_solve`), with tests and benchmark coverage.
43. Sparse iterative APIs now include explicit factorization-reuse entry points for repeated RHS solves (`*_with_factorization` variants for `gmres_ilut`, `bicgstab_ilu0`, `bicgstab_ilut`) plus config-driven `bicgstab_ilut_solve_with_config`, with parity tests and benchmark visibility.
44. Sparse iterative breadth now also includes ILU(0)-preconditioned GMRES (`gmres_ilu0_solve` and reuse variant), with parity/error tests and benchmark visibility for setup-vs-solve paths.
45. Copy-elision/performance-contract audit (`N-061`) is complete: remaining avoidable algorithm-internal owned materializations were removed in iterative/QR/SVD paths, and unavoidable allocations are now explicitly documented in `docs/PERFORMANCE_CONTRACTS.md`.
46. Complex parity across major algorithms is now complete: matrix primitives and orthogonalization gained complex APIs, iterative solvers gained complex CG/GMRES, and tensor gained complex cube and last-axis tensor operations with parity tests.
47. Sparse factorization depth now includes ILDL(0) and ILDL0-preconditioned GMRES/BiCGSTAB (direct + factorization-reuse APIs), with unit tests and sparse benchmark coverage.
48. Tensor higher-rank algebra baseline now includes axis permutation, explicit-axis contraction, and N-D batched last-two matmul across real/complex APIs, with unit tests and tensor benchmark coverage.
49. Accelerator baseline now includes compile-time backend dispatch over CPU/GPU targets, with dedicated accelerator benchmark coverage in smoke/report pipelines.
50. Accelerator CPU kernel depth includes deterministic chunking/tile-style partition helpers with explicit validation and coverage in tests/benchmarks.
51. Sparse factorization depth now includes ILU(k) (`ILUKConfig`, factorization/apply APIs) with ILUK-preconditioned GMRES/BiCGSTAB (direct, factorization-reuse, and multi-RHS workflows), plus sparse benchmark coverage.
52. Sparse depth now includes direct sparse LU factorization workflows (`sparse_lu_factor`, direct/reuse/multi-RHS solve paths) with unit and benchmark coverage.
53. Tensor depth now includes rank-3 HOSVD (`hosvd3`, reconstruction) and binary einsum ergonomics for real and complex tensors.
54. Accelerator depth now includes feature-gated concrete GPU `f32` matmat execution via `accelerator-wgpu`.
55. Batched decomposition-level workflows are now first-class (`nabled-linalg::batched`) for QR/SVD/LU/Cholesky/symmetric eigen.
56. Dense batched broadcast semantics now include broadcast-left/right matrix products over batch stacks.
57. Non-symmetric eigen depth now includes balancing APIs and matched left/right eigenvector outputs.
58. Advanced optimization breadth now includes constrained (`projected_gradient_descent_box`), stochastic (`stochastic_gradient_descent`), and quasi-Newton (`bfgs`) methods.
59. Accelerator internals are now modularized (`accelerator.rs` + `accelerator/*`) with backend-specific kernel files and per-operation trait dispatch (`MatMatKernel<T>`), preserving public API while improving scalability for future GPU and multi-node expansion.
60. Execution-model terminology is now locked and documented as orthogonal axes: `Provider` (decomposition source), `Backend` (kernel execution target), and `Kernel` (operation-family contract).
61. Accelerator kernel dispatch has expanded to matrix-vector operations (`MatVecKernel<T>` + backend dispatch entrypoints), while provider selection remains compile-time via feature-gated domain paths.
62. First benchmark-driven optimization loop has materially reduced dense-kernel overhead: matrix/vector hot paths now use ndarray optimized kernels (`dot`, `general_mat_mul`) in place of manual loop baselines, bringing nabled matrix/vector smoke benchmarks to near-competitor parity.
63. Kernel-dispatch coverage has expanded beyond the initial slice: per-operation traits now include batched dense matmat and sparse matvec, and stable allocating matrix/sparse APIs are wired through compile-time backend dispatch.
64. The interim CPU-sharding backend has been removed; backend execution targets are now explicitly `CpuBackend` and `CudaBackend`, with multi-node support deferred to a future backend.
65. Full v1 kernel-family scope is now explicitly cataloged in `docs/KERNEL_CATALOG.md` so kernelization work is deterministic and auditable.
66. Kernel-model wiring for the current v1 catalog is complete: dense/sparse/vector/tensor/triangular kernel families are all wired through compile-time dispatch with backend coverage tests.
67. V1 stability contract is now explicit in `docs/V1_STABILITY.md`, including required tensor/GPU surface, mixed execution behavior, feature matrix, and no-surprises audit criteria.
68. Accelerator v1 GPU contract depth now includes tested tensor batched-last-two GPU parity (`f32`) and explicit tested strict-or-fallback behavior for out-of-scope CUDA tensor kernels/dtypes.
69. Local/CI quality gates now enforce accelerator feature permutations (`accelerator-rayon`, `accelerator-wgpu`, and provider combinations), not only default/provider paths.
70. Higher-level ML/stat complex parity (`B-P1-006`) is now complete across `stats`, `regression`, `pca`, and `optimization`, closing the last declared v1 capability blocker.

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

1. Resolve `K-*` architecture decisions (starting with `K-006`, then `K-004`/`K-008` alignment).
2. Run benchmark/regression optimization passes (outlier triage, allocation audit, SIMD/threading opportunities).
3. Keep execution updates current in `docs/EXECUTION_TRACKER.md`.

## Completion Criteria For Migration

1. Facade package is `crates/nabled` and root manifest is virtual-workspace-only.
2. Domain modules live in `crates/nabled-linalg` and `crates/nabled-ml`.
3. CI, benches, examples, and tests run workspace-wide without root-implementation coupling.
