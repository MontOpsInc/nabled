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
12. New benchmark suites now cover `sparse`, `schur`, `sylvester`, and `optimization`.
13. Complex decomposition parity started (`QR` complex path and provider-backed complex SVD).
14. Shared cross-domain taxonomy exists via `nabled-core::errors::NabledError`.
15. Complex-number vector parity is now present (Hermitian dot, complex norm/cosine).
16. First-order optimization primitives are now present (line search, gradient descent, Adam).
17. View-first (`ArrayView*`) API coverage is expanded across remaining heavy linalg/ml domains with parity tests.
18. Public API namespaces are flattened; `ndarray_*` wrapper modules are removed in favor of direct domain APIs.

## Current Code Ownership

1. `crates/nabled-core`
   - shared prelude, validation, and core error scaffolding.
2. `crates/nabled-linalg`
   - decomposition, solver, and matrix-function domains:
     `svd`, `qr`, `lu`, `cholesky`, `eigen`, `schur`, `polar`, `sylvester`,
     `matrix_functions`, `orthogonalization`, `triangular`, `vector`.
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

1. Expand complex-number parity beyond QR/SVD and vector kernels.
2. Add explicit tests for `IntoNabledError` mappings and maintain mapping stability.
3. Add benchmark competitor baselines for newly added benchmark domains.
4. Keep execution updates current in `docs/EXECUTION_TRACKER.md`.

## Completion Criteria For Migration

1. Facade package is `crates/nabled` and root manifest is virtual-workspace-only.
2. Domain modules live in `crates/nabled-linalg` and `crates/nabled-ml`.
3. CI, benches, examples, and tests run workspace-wide without root-implementation coupling.
