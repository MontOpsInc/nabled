# Status Snapshot

Last updated: 2026-02-28

## Summary

Workspace migration for library domains is complete.

1. Workspace members exist: `nabled-core`, `nabled-linalg`, `nabled-ml`.
2. `crates/nabled` is the facade package re-exporting workspace crates.
3. `crates/nabled/src/` contains facade/library entrypoint and binary tooling only.

## Current Code Ownership

1. `crates/nabled-core`
   - shared prelude, validation, and core error scaffolding.
2. `crates/nabled-linalg`
   - decomposition, solver, and matrix-function domains:
     `svd`, `qr`, `lu`, `cholesky`, `eigen`, `schur`, `polar`, `sylvester`,
     `matrix_functions`, `orthogonalization`, `triangular`.
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

## Next Required Milestone

Harden workspace contracts and release readiness:

1. Consolidate shared trait/error patterns in `nabled-core`.
2. Continue benchmark/regression hardening by domain.
3. Finalize facade API boundaries in `crates/nabled` and remove stale compatibility shims as requested.

## Completion Criteria For Migration

1. Facade package is `crates/nabled` and root manifest is virtual-workspace-only.
2. Domain modules live in `crates/nabled-linalg` and `crates/nabled-ml`.
3. CI, benches, examples, and tests run workspace-wide without root-implementation coupling.
