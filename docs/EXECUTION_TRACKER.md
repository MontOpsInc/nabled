# Execution Tracker

Last updated: 2026-02-28

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

## Next

1. `N-006` (P0): Replace provider fallback stubs with provider-native kernels for QR/LU/Cholesky/Eigen.
2. `N-007` (P0): Expand `_into`/workspace APIs through remaining heavy domains (`matrix_functions`, `schur`, `sylvester`).
3. `N-008` (P0): Add sparse primitives and solvers baseline (`sparse matrix/vector`, sparse solves).
4. `N-009` (P0): Add complex-number parity for Tier-A kernels.
5. `N-010` (P0): Add optimization primitives beyond CG/GMRES (line search + first-order optimizers).

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
