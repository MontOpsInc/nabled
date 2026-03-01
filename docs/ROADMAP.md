# Roadmap

## North Star

`nabled` becomes the ndarray-native, production-grade linear algebra and ML foundation for high-performance vector and matrix workflows.

## Phase 1: Workspace and Policy Reset

1. Convert repository to Cargo workspace.
2. Create `nabled-core`, `nabled-linalg`, `nabled-ml`, and facade crate `nabled`.
3. Move CI, benchmarks, and `just` commands to workspace-aware execution.
4. Keep existing functionality green during structural migration.

Current status:
1. Workspace members for `nabled-core`, `nabled-linalg`, and `nabled-ml` are in place.
2. Domain implementations have been relocated into `nabled-linalg` and `nabled-ml`.
3. `crates/nabled` now acts as a facade over workspace crates.
4. See `docs/STATUS.md` for the current migration truth.

Exit criteria:
1. Root is a virtual workspace manifest and `crates/nabled` is facade-only. ✅
2. Algorithm implementations no longer live in a root `src/*.rs` tree. ✅
3. Workspace crates are the primary implementation units. ✅

## Phase 2: Ndarray-Only Surface

1. Remove non-ndarray public APIs.
2. Normalize APIs around ndarray vectors, matrices, and views.
3. Rework examples/tests/benchmarks to ndarray-first usage.
4. Keep root-to-crate re-exports explicit and minimal.

## Phase 3: Backend Feature Contract and Dispatch Cleanup

1. Introduce and stabilize feature contract:
   - `blas` baseline
   - provider feature(s), starting with `openblas-system`.
2. Remove hardcoded OS gates from LAPACK paths; prefer feature-gated selection.
3. Unify public API entrypoints so backend choice is internal (no public `*_lapack` duplication).
4. Apply one consistent module pattern for internal/provider implementation splits.

Current status:
1. Feature contract and API surface cleanup are complete.
2. Dense module dispatch and naming consistency pass is complete for Tier-A kernels.
3. Remaining backend-depth work is concentrated in selected secondary domains and broader provider parity.

## Phase 4: Kernel Realignment

1. Port each domain to ndarray-native compute paths.
2. Eliminate hidden conversion paths in algorithm implementations.
3. Validate numerical correctness with targeted regression tests.

Suggested order:
1. `svd`, `qr`, `lu`, `cholesky`, `eigen`, `triangular` ✅ API contract/alignment complete
2. `schur`, `sylvester`, `polar`, `matrix_functions`
3. `pca`, `regression`, `iterative`, `jacobian`

## Phase 5: Production Hardening

1. Stabilize error model and API contracts.
2. Expand property/invariant tests and adversarial cases.
3. Establish performance baselines and regression thresholds by domain.
4. Preserve coverage >= 90% with meaningful tests.

## Phase 6: Readiness

1. Final docs pass for end users and contributors.
2. Publish readiness checklist and release criteria.
3. Freeze v0 API candidate and run full validation.
