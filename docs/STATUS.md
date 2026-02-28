# Status Snapshot

Last updated: 2026-02-28

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

## Next Required Milestone

Harden workspace contracts and release readiness:

1. Implement native provider kernels (not fallback stubs) for QR/LU/Cholesky/Eigen.
2. Expand allocation-controlled APIs to remaining heavy kernels (`matrix_functions`, `schur`, `sylvester`).
3. Add sparse/vectorized batch domains per `docs/CAPABILITY_MATRIX.md`.
4. Continue benchmark/regression hardening by domain and operation class.
5. Track active execution state in `docs/EXECUTION_TRACKER.md`.

## Completion Criteria For Migration

1. Facade package is `crates/nabled` and root manifest is virtual-workspace-only.
2. Domain modules live in `crates/nabled-linalg` and `crates/nabled-ml`.
3. CI, benches, examples, and tests run workspace-wide without root-implementation coupling.
