# Execution Tracker

Last updated: 2026-03-03

## Purpose

This is the operational companion to `docs/CAPABILITY_MATRIX.md`.

Use this file to resume work quickly after context compaction without re-auditing the full codebase.

## Usage Rules

1. Treat this file as the canonical `Done / Next / Needed` tracker.
2. Update it in the same change set when non-trivial implementation work lands.
3. Only do a full repository re-assessment if:
   - this file is stale,
   - statuses conflict with observed code, or
   - architectural direction changed.

## Current Baseline

1. Workspace migration is complete (`nabled-core`, `nabled-linalg`, `nabled-ml`, `nabled` facade).
2. CI and local quality gates are workspace-aware and passing.
3. Capability matrix exists and is the scope/gap source of truth.
4. macOS OpenBLAS environment wiring is centralized in `.justfile` for provider-enabled recipes.
5. Quality gates continue to pass in both internal and provider-enabled modes after capability expansion.
6. Benchmark smoke/report coverage now includes matrix/tensor suites in addition to prior dense/sparse suites.
7. Ordered v1 stability gate blockers are closed; focus is now K-series normalization and benchmark/performance hardening passes.

## V1 Stability Gate (Ordered, Required)

`nabled` is considered **100% stable for v1** only when all items below are complete, in order:

1. Required tensor API surface is fully explicit and complete (owned/view/into, real/complex where applicable, stable allocation semantics).
2. Required GPU kernel surface is concrete (no placeholder behavior for in-scope kernels) and documented by dtype/operation support.
3. Mixed execution paths are deterministic and unsurprising (provider/backend/kernel combinations have explicit behavior and docs).
4. Required feature/build matrix is fully exercised in CI and local checks (including provider-enabled and GPU-enabled paths where applicable).
5. Complex parity is complete for higher-level ML/statistical domains (`stats`, `regression`, `pca`, `optimization`) with explicit real/complex API contracts.
6. Final no-surprises audit passes (allocation contracts, error semantics, fallback rules, and docs all aligned with behavior).

## Done

1. `D-001`: Workspace architecture established and stabilized.
2. `D-002`: Root moved to virtual workspace manifest with crate-level facade at `crates/nabled`.
3. `D-003`: CI and `.justfile` migrated to workspace/package-aware commands.
4. `D-004`: Strict lint/test/doc checks passing after migration.
5. `D-005`: Capability matrix documented with sufficiency verdict and ordered execution path.
6. `D-006`: Homebrew OpenBLAS env setup is now recipe-driven for macOS provider-enabled commands.
7. `D-007`: Feature contract migrated to `blas` + `openblas-system` across workspace crates.
8. `D-008`: Provider code paths are now feature-gated without hardcoded OS cfg branches.
9. `D-009`: Public `*_lapack` API duplication removed; backend choice stays internal.
10. `D-010`: `just checks` and CI now validate internal mode and provider-enabled mode.
11. `D-011`: Dense kernels (`svd`, `qr`, `lu`, `cholesky`, `eigen`) now share one dispatch contract and normalized API names.
12. `D-012`: `nabled-linalg::vector` introduced with dot/norm/cosine/pairwise/batched primitives.
13. `D-013`: Explicit allocation/workspace API pattern established (`*_into`, reusable workspace structs).
14. `D-014`: Tier-A benchmark suites expanded (LU, Cholesky, Eigen, Vector).
15. `D-015`: Benchmark reporting/classification updated for expanded domain coverage.
16. `D-016`: Dense provider paths are now native for QR/LU/Cholesky/Eigen hot operations (no provider stubs in those paths).
17. `D-017`: `_into` + reusable workspace APIs expanded across `matrix_functions`, `schur`, and `sylvester`.
18. `D-018`: Sparse baseline landed with `CSR` primitives (`matvec`, `matvec_into`) and Jacobi sparse solve.
19. `D-019`: Complex-number parity baseline added for vector kernels (`dot_hermitian`, complex norm/cosine).
20. `D-020`: First-order optimization primitives added in `nabled-ml` (Armijo backtracking, gradient descent, Adam).
21. `D-021`: Benchmark suites expanded for `sparse`, `schur`, `sylvester`, and `optimization`; smoke recipes and benchmark classifier updated.
22. `D-022`: Complex parity extended into decomposition APIs (`QR` complex decomposition and provider-backed complex SVD).
23. `D-023`: Sparse roadmap advanced with `COO` support, COO→CSR conversion, and Gauss-Seidel sparse solve.
24. `D-024`: View-first API pass started on hot paths (`vector`, `qr`, `svd`) via `ArrayView*` entrypoints.
25. `D-025`: Cross-domain error taxonomy consolidated in `nabled-core` via `NabledError` + `IntoNabledError` mappings in linalg/ml crates.
26. `D-026`: Scoped line coverage restored above gate (`90.63%`) with targeted tests across linalg/ml/core low-coverage paths.
27. `D-027`: Coverage policy aligned in local/CI: shared ignore regex (`errors|examples|benches|src/bin`), `--lib --tests` scope, and hard `>=90%` gate.
28. `D-028`: View-first API pass expanded across remaining heavy ndarray APIs (`cholesky`, `lu`, `eigen`, `matrix_functions`, `orthogonalization`, `polar`, `schur`, `sylvester`, `triangular`, `stats`, `regression`, `pca`) with parity tests.
29. `D-029`: Flattened public API namespaces by removing `ndarray_*` wrapper modules across linalg/ml domains; public call surface is now direct per-domain (`nabled::cholesky::decompose`, etc.) and all workspace call sites (tests/benches/examples) were migrated.
30. `D-030`: Complex parity expanded beyond QR/SVD into polar decomposition and triangular solves.
31. `D-031`: Dedicated `IntoNabledError` mapping-stability tests added for `nabled-linalg` and `nabled-ml`.
32. `D-032`: Competitor benchmark baselines added where practical (`vector`, `sparse`, `optimization`) with report classifier support.
33. `D-033`: Low per-file coverage targets completed for `eigen`, `lu`, `orthogonalization`, and `polar`.
34. `D-034`: Dedicated benchmark suites added for `polar` and `orthogonalization`, with smoke/report wiring and classifier support.
35. `D-035`: Practical manual competitor baselines added for `schur` and `sylvester`.
36. `D-036`: Allocation-transparency docs completed for allocating `*_view`/convenience wrappers across linalg/ml modules.
37. `D-037`: Complex parity expanded into LU/Cholesky via provider-backed complex solve/inverse/decomposition paths with explicit no-provider behavior.
38. `D-038`: Complex parity expanded into Schur and Sylvester/Lyapunov with provider-backed complex APIs, view/into/workspace variants, and explicit no-provider behavior.
39. `D-039`: Complex matrix-functions parity baseline added (complex `matrix_exp*` Taylor path and provider-backed complex `matrix_log_svd*` with owned/view/into/workspace APIs and feature-split tests).
40. `D-040`: Complex matrix-functions parity expanded to Hermitian eigen-based `exp/log/power/sign` paths (owned/view/into/workspace), with explicit no-provider behavior where provider-backed eigendecomposition is required.
41. `D-041`: `N-024A` complete for currently identified dense domains.
42. `D-042`: `N-024B` internal/no-provider complex parity advanced across LU, Cholesky, Schur, Sylvester/Lyapunov, and Polar; complex no-provider behavior now succeeds in these domains instead of provider-only erroring.
43. `D-043`: Internal complex SVD path added for no-provider builds (Schur-derived over `A^H A`), and complex `matrix_log_svd*` no-provider behavior now succeeds.
44. `D-044`: Internal complex Hermitian-eigen matrix-functions paths landed (`log/power/sign`), completing `N-024B` for currently implemented complex-capable dense domains.
45. `D-045`: `N-024C` integration coverage landed for complex parity and error mapping in facade integration tests (`test_complex_dense_parity_pipeline`, `test_complex_error_mapping_paths`) and now runs in both internal/provider CI jobs.
46. `D-046`: `N-024C` benchmark visibility expanded with complex benchmark cases in `svd_benchmarks`, `matrix_functions_benchmarks`, and `polar_benchmarks` (smoke-validated).
47. `D-047`: `N-025` completed: dense-kernel tolerance/iteration defaults are now centralized in `internal::DenseKernelPolicy` and applied consistently across `svd`, `qr`, `eigen`, `schur`, `polar`, `matrix_functions`, `lu`, and `cholesky` paths.
48. `D-048`: Dense pipeline module expanded: `nabled-linalg::matrix` now includes batched matrix-matrix kernels with owned/view/into APIs and tests.
49. `D-049`: Sparse breadth expanded: CSC format + CSR↔CSC conversion, sparse-sparse multiplication, and `BiCGSTAB` solver added with coverage.
50. `D-050`: Optimization breadth expanded with momentum descent and `RMSProp` (configs, APIs, and tests).
51. `D-051`: Non-symmetric eigen support broadened with provider-backed `Eig` paths and stabilized internal complex small-matrix handling.
52. `D-052`: P2 baselines landed: `nabled-linalg::tensor` (cube primitives) and `nabled-linalg::accelerator` (compile-time backend contracts).
53. `D-053`: Shared taxonomy and facade/integration coverage updated for new matrix/sparse/tensor/accelerator capabilities.
54. `D-054`: Copy-elision pass advanced in ML domains: `stats`, `regression`, and `pca` view APIs now route through view-native internals without hidden `to_owned()` materialization in key paths.
55. `D-055`: Sparse depth expanded with Jacobi preconditioning and `PCG` solve (`jacobi_preconditioner`, `apply_jacobi_preconditioner`, `pcg_solve`) plus coverage.
56. `D-056`: Tensor breadth expanded from rank-3 baseline with higher-rank `ArrayD` last-axis operations (`sum`, `l2_norm`, `normalize`, `batched_dot`) plus coverage.
57. `D-057`: Accelerator domain now includes explicit serial matmat and feature-gated accelerated matmat kernel (`accelerator-rayon`) with stable error mapping and integration coverage.
58. `D-058`: Benchmark/report pipeline expanded with `matrix_benchmarks` and `tensor_benchmarks`, sparse benchmark additions, and benchmark-report classifier mappings for new suites.
59. `D-059`: Sparse factorization-grade baseline advanced with ILU(0): `ilu0_factor`, `apply_ilu0_preconditioner`, and `bicgstab_ilu0_solve`, plus tests and sparse benchmark coverage.
60. `D-060`: Copy-elision pass expanded through core linalg view APIs (`orthogonalization`, `qr`, `svd`, `schur`, `sylvester`), with view-native dispatch paths and removed wrapper-level `to_owned()` materialization in no-provider mode.
61. `D-061`: Copy-elision pass expanded through additional core linalg domains (`lu`, `cholesky`, `eigen`, `triangular`) with view-native validation/dispatch and no wrapper-level `to_owned()` in no-provider view paths.
62. `D-062`: Copy-elision pass extended through remaining convenience view wrappers in `polar` and `pca`, and provider-side view dispatch in `lu`/`cholesky`/`eigen` now avoids wrapper-level owned materialization.
63. `D-063`: Sparse factorization depth expanded with IC(0): `ic0_factor`, `apply_ic0_preconditioner`, and `pcg_ic0_solve`, with tests and sparse benchmark coverage.
64. `D-064`: Sparse factorization depth expanded with ILUT: `ilut_factor`, `apply_ilut_preconditioner`, and `bicgstab_ilut_solve`, with tests and sparse benchmark coverage.
65. `D-065`: ILUT ergonomics and solver breadth expanded with `ILUTConfig` policy helpers, config-driven factorization/solve APIs, and ILUT-preconditioned `gmres` sparse solve (`gmres_ilut_solve` / `gmres_ilut_solve_with_config`), plus tests and benchmark coverage.
66. `D-066`: Sparse solve reuse APIs added for factorization-caching workflows (`gmres_ilut_solve_with_factorization`, `bicgstab_ilu0_solve_with_factorization`, `bicgstab_ilut_solve_with_factorization`, `bicgstab_ilut_solve_with_config`) with parity tests and sparse benchmark coverage for reuse paths.
67. `D-067`: Sparse solver breadth expanded with ILU(0)-preconditioned `gmres` (`gmres_ilu0_solve`, `gmres_ilu0_solve_with_factorization`) plus dimension/error parity tests and benchmark coverage (`gmres_ilu0_solve`, `gmres_ilu0_solve_reuse`).
68. `D-068`: `N-061` copy-elision audit completed: removed remaining avoidable algorithm-internal materializations in iterative/QR/SVD paths, documented unavoidable allocations in `docs/PERFORMANCE_CONTRACTS.md`, and promoted P0 performance-contract status to implemented.
69. `D-069`: `B-P1-003` complex-parity closure for major algorithms: complex matrix primitives, complex orthogonalization, complex iterative solvers (CG/GMRES), and complex tensor kernels/last-axis ops are now implemented with parity tests in internal/provider-aware quality gates.
70. `D-070`: Sparse depth advanced with ILDL(0) (`ildl0_factor`, `apply_ildl0_preconditioner`) and ILDL0-preconditioned GMRES/BiCGSTAB (including factorization-reuse APIs), with tests and sparse benchmark coverage.
71. `D-071`: Tensor higher-rank algebra baseline expanded with axis permutation (`permute_axes*`), explicit-axis contraction (`contract_axes*`), and N-D batched last-two matmul (`batched_matmul_last_two*`) across real/complex APIs, plus tests and tensor benchmark coverage.
72. `D-072`: Accelerator baseline advanced with row-sharded CPU matmat experiments, expanded error mapping, integration/unit coverage, and new `accelerator_benchmarks` wired into smoke/report pipelines.
73. `D-073`: Accelerator CPU-tiling experiments were added with validation/error contracts and benchmark visibility.
74. `D-074`: Sparse factorization depth expanded with ILU(k) (`iluk_factor`, `apply_iluk_preconditioner`, `ILUKConfig`) and ILUK-preconditioned GMRES/BiCGSTAB (including factorization-reuse/multi-RHS APIs), plus tests and sparse benchmark coverage.
75. `D-075`: `N-060` sparse depth continued with direct sparse LU workflows (`sparse_lu_factor`, direct/reuse/multi-RHS solve APIs), unit coverage, and sparse benchmark smoke/report cases for setup-vs-reuse visibility.
76. `D-076`: `N-062` tensor depth continued with rank-3 HOSVD (`hosvd3` + reconstruction) and two-operand einsum ergonomics for real/complex tensors, with parity/error tests.
77. `D-077`: `N-063` accelerator depth continued with CPU scheduling experiments and feature-gated concrete GPU `f32` matmat kernel execution path (`accelerator-wgpu`) plus provider-aware tests.
78. `D-078`: `B-P1-001` completed with batched decomposition-level APIs (`nabled-linalg::batched` for QR/SVD/LU/Cholesky/symmetric eigen) and richer dense broadcast semantics (`batched_matmat_broadcast_left/right` owned/view/into variants).
79. `D-079`: `B-P1-002` closed for current scope with direct sparse LU factorization-grade workflows (factor/reuse/multi-RHS), extending preconditioned iterative sparse depth into direct solve pipelines.
80. `D-080`: `B-P1-004` completed with non-symmetric eigen depth improvements: balancing APIs (`balance_nonsymmetric*`) and matched left/right eigenvector surface (`nonsymmetric_bi*`) with configuration controls.
81. `D-081`: `B-P1-005` completed with advanced optimization breadth (`projected_gradient_descent_box`, `stochastic_gradient_descent`, `bfgs`) plus config validation and convergence/error tests.
82. `D-082`: Accelerator architecture refactored into modern module layout (`accelerator.rs` + `accelerator/*`), separating backend markers and backend dispatch with per-operation kernel traits (`MatMatKernel<T>`) to keep future accelerator expansion composable.
83. `D-083`: Documentation model clarified and locked around orthogonal execution axes (`Provider / Backend / Kernel`) with corresponding tracker follow-on items (`K-006..K-008`).
84. `D-084`: Initial execution-axis normalization landed in code: expanded backend kernel dispatch for matrix-vector operations (`MatVecKernel<T>`, backend dispatch entrypoints, and tests) while keeping provider selection compile-time in domain modules.
85. `D-085`: Benchmark-driven optimization loop executed for dense matrix/vector hot paths: `matrix` and CPU-kernel multiply paths now use ndarray optimized kernels (`dot`/`general_mat_mul`) instead of manual triple loops, and vector scalar kernels now use optimized `dot`-based implementations; benchmark reruns show matrix/vector parity moved from major regressions to near-equal competitor ranges in smoke sizes.
86. `D-086`: `K-007` kernel expansion advanced: added `BatchedMatMatKernel<T>` and `SparseMatVecKernel`, implemented backend dispatch for CPU/CUDA-placeholder, added coverage tests, and wired stable allocating matrix/sparse APIs (`matrix::matvec`, `matrix::matmat`, `matrix::batched_matmat`, `sparse::matvec`) through compile-time backend dispatch.
87. `D-087`: Kernel model scope is now explicitly locked in `docs/KERNEL_CATALOG.md`; full v1 kernel-family inventory and status (`Wired/Dispatch/Contract`) is documented with orchestration baseline rules for `K-008`.
88. `D-088`: `K-007` is now complete for the v1 kernel catalog: remaining kernel families are wired end-to-end (CPU dispatch + CUDA placeholders where applicable) across dense/sparse/vector/tensor/triangular APIs, with dispatch tests and full quality-gate verification.
98. `D-098`: Removed the interim CPU-sharding backend and all related API/bench/docs surface so execution targets are now backend-pure (`CpuBackend`, `CudaBackend`) with explicit multi-node work deferred to a future backend.
89. `D-089`: GPU baseline capability expanded beyond single-op matmul: `CudaBackend` now supports `f32` matvec and batched matmat dispatch via `wgpu`, and tensor batched-last-two `f32` dispatch is backed by GPU matmul composition.
90. `D-090`: Tensor higher-rank API ergonomics expanded with explicit view-first entrypoints (`*_view`, `*_view_into`) for last-axis reductions, axis contractions, and N-D batched last-two matmul across real and complex paths.
91. `D-091`: Allocation-contract regressions introduced by kernel routing were corrected (`*_into` paths in vector/sparse/triangular/tensor no longer allocate hidden temporaries).
92. `D-092`: `V1-G1` closed: required tensor v1 surface is now explicitly locked in `docs/V1_STABILITY.md` and validated with added owned/view/into parity tests across real and complex `ArrayD` contraction/reduction/matmul families.
93. `D-093`: `V1-G2` closed: required GPU v1 kernel surface is now explicitly locked in `docs/V1_STABILITY.md`, with tested parity for supported `f32` CUDA-dispatched paths and tested strict-or-fallback behavior for out-of-scope CUDA tensor kernels/dtypes.
94. `D-094`: `V1-G3` closed: tensor view-first parity/invariant coverage expanded for required v1 families (`sum_last_axis*`, `contract_axes*`, `batched_matmul_last_two*`, real+complex).
95. `D-095`: `V1-G4` closed: GPU tensor coverage expanded with CPU-vs-GPU parity testing (`f32`) and strict-or-fallback behavior assertions (`f64`/other out-of-scope tensor CUDA kernels).
96. `D-096`: `V1-G5` closed: CI/local quality gates now enforce provider/backend/kernel permutations including `accelerator-rayon`, `accelerator-wgpu`, and provider+accelerator combinations.
97. `D-097`: `V1-G6` closed: no-surprises audit signoff documented and synchronized across `docs/V1_STABILITY.md`, `docs/PERFORMANCE_CONTRACTS.md`, `docs/CAPABILITY_MATRIX.md`, and `docs/STATUS.md`.
99. `D-099`: v1 readiness contract was tightened: higher-level ML/stat complex parity was promoted from post-v1 backlog into required v1 scope (`B-P1-006`), reopening the v1 gate until implemented.
100. `D-100`: `B-P1-006` is complete: higher-level ML/stat complex parity is now implemented in `nabled-ml` (`stats`, `regression`, `pca`, `optimization`) with owned/view APIs and coverage tests.

## Next

1. `K-006`: Lock module-level normalization plan for `Provider / Backend / Kernel` naming and ownership boundaries.
2. `K-005`: Start benchmark-driven optimization pass (outlier triage, allocation audit, SIMD/threading opportunities).
3. `K-004`: Decide exact provider expansion policy beyond `openblas-system`.
4. `K-008`: Continue kernel orchestration cleanup where API-level dispatch still has ad hoc pathways.

## Needed

1. `K-006`: Module ownership boundary lock for Provider/Backend/Kernel axes.
2. `K-005`: Outlier-ranked benchmark optimization plan + execution log.
3. `K-004`: Provider expansion decision record beyond `openblas-system`.
4. `K-008`: Kernel orchestration cleanup plan + implementation log for remaining ad hoc API dispatch paths.

## Backlog (From Capability Matrix)

1. Advanced tensor algebra depth beyond the v1 baseline.
2. Broader GPU dtype/op coverage and future multi-node backend breadth.

## Resume Protocol (Compaction-Friendly)

1. Read in this order:
   - `docs/README.md`
   - `docs/DECISIONS.md`
   - `docs/CAPABILITY_MATRIX.md`
   - `docs/KERNEL_CATALOG.md`
   - `docs/PERFORMANCE_CONTRACTS.md`
   - `docs/V1_STABILITY.md`
   - `docs/EXECUTION_TRACKER.md`
   - `docs/STATUS.md`
2. Start from the highest-priority open `N-*` item unless maintainers redirect.
3. Keep item IDs in PR/commit notes when relevant so progression stays auditable.
