# Execution Tracker

Last updated: 2026-03-01

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

## Next

1. `N-018` (P0): Extend complex parity to additional decomposition/solver domains beyond QR/SVD.
2. `N-019` (P0): Add dedicated tests for `IntoNabledError` conversion coverage and mapping stability.
3. `N-020` (P0): Add benchmark competitor baselines for new benchmark suites where practical.
4. `N-021` (P1): Incrementally raise low per-file coverage in `eigen`, `lu`, `orthogonalization`, and `polar` while keeping API behavior unchanged.

## Needed

1. `K-001`: Final API shape decision for view-first signatures (`ArrayView*`) versus owned-only signatures.
2. `K-002`: Decision on standardized workspace type pattern (per-domain workspace vs shared core workspace).
3. `K-003`: Priority order for sparse matrix formats and sparse solver entrypoints.
4. `K-004`: Decide exact provider expansion policy beyond `openblas-system`.

## Backlog (From Capability Matrix)

1. `B-P1-001`: Batched operations over many vectors/matrices (beyond current vector-kernel baseline).
2. `B-P1-002`: Sparse matrix/vector primitives and solver coverage.
3. `B-P1-003`: Complex-number parity across major algorithms.
4. `B-P1-004`: Broader non-symmetric dense eigen capabilities.
5. `B-P1-005`: Expanded optimization primitives.

## Resume Protocol (Compaction-Friendly)

1. Read in this order:
   - `docs/README.md`
   - `docs/DECISIONS.md`
   - `docs/CAPABILITY_MATRIX.md`
   - `docs/EXECUTION_TRACKER.md`
   - `docs/STATUS.md`
2. Start from the highest-priority open `N-*` item unless maintainers redirect.
3. Keep item IDs in PR/commit notes when relevant so progression stays auditable.
