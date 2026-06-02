# Nabled Docs

This folder contains the current, authoritative direction for `nabled`.

## Current Direction

1. `nabled` is becoming an `ndarray`-first library.
2. Interop between Rust linalg libraries is not a current objective.
3. Performance, correctness, and composability are first-class requirements.
4. Workspace architecture is the foundation for scale.
5. Backend policy is compile-time feature selection (internal default, provider-enabled acceleration).

## Documents

1. `docs/DECISIONS.md`: locked decisions and constraints.
2. `docs/CAPABILITY_MATRIX.md`: capability inventory, gap analysis, and scope verdict.
3. `docs/REFERENCE_RUBRIC.md`: external-anchor parity rubric and objective "done"/production-readiness criteria.
4. `docs/BENCHMARK_TRACKER.md`: chunked benchmark audit, current results, and optimization loop tracking.
5. `docs/GPU_V2_TRACKER.md`: v2 GPU/provider expansion tracker (batch policy, MAGMA integration, remote verification).
6. `docs/MAGMA_SIGNOFF.md`: MAGMA route signoff matrix with per-API execution evidence and pending closure rows.
7. `docs/REMOTE_GPU_WORKFLOW.md`: canonical remote NVIDIA workflow (image + tmux-first execution scripts).
8. `docs/KERNEL_CATALOG.md`: complete kernel contract inventory, wiring status, and done-state definition.
9. `docs/PERFORMANCE_CONTRACTS.md`: allocation/copy contract and copy-elision audit status.
10. `docs/V1_STABILITY.md`: explicit v1-stability contract (required tensor/GPU surface, feature matrix, no-surprises signoff).
11. `docs/EXECUTION_TRACKER.md`: current `Done / Next / Needed` plan for compaction-safe continuation.
12. `docs/architecture.md`: target package/module architecture.
13. `docs/ROADMAP.md`: implementation sequencing and milestones.
14. `docs/PUBLISH_CHECKLIST.md`: crates.io publish gate and release-day checklist.
15. `docs/PYPI_PUBLISH.md`: **pynabled** PyPI releases (tags, CI wheel smoke, `just tag-pypi-release`).
16. `docs/PYNABLED_GAPS_AUDIT.md`: Python-branch audit findings and merge blockers for `feat/pynabled-bindings`.
17. `docs/PYNABLED_PARITY_MATRIX.md`: authoritative Rust-vs-Python parity target for `pynabled` release readiness.
18. `docs/PYNABLED_ARCHITECTURE.md`: canonical Python carrier, copy/allocation, callback, and result-fidelity contract for `pynabled`.
19. `docs/STATUS.md`: current migration snapshot (source of truth for what has been moved).
20. `docs/NARROW_INTEROP.md`: task workload for narrow (Arrow-ndarray bridge) interop improvements.
21. `docs/NDARROW_INTEGRATION.md`: facade-level Arrow interop contract and current adapter surface.
22. `docs/ARROW_SUPPORT_MATRIX.md`: direct Arrow-ingress coverage against the current public API.
23. `docs/EMBEDDINGS.md`: embedding retrieval compute (`nabled-embeddings`), LanceDB plug-in stance, Rust/Python/Arrow surface, and non-goals.
24. `docs/BENCHMARKS.md`: embeddings benchmark methodology and honest exact-vs-exact framing.

## Context Resume Protocol

When starting from a compacted/partial context, read documents in this order:

1. `docs/README.md`
2. `docs/DECISIONS.md`
3. `docs/CAPABILITY_MATRIX.md`
4. `docs/REFERENCE_RUBRIC.md`
5. `docs/BENCHMARK_TRACKER.md`
6. `docs/GPU_V2_TRACKER.md`
7. `docs/MAGMA_SIGNOFF.md`
8. `docs/REMOTE_GPU_WORKFLOW.md`
9. `docs/KERNEL_CATALOG.md`
10. `docs/PERFORMANCE_CONTRACTS.md`
11. `docs/V1_STABILITY.md`
12. `docs/EXECUTION_TRACKER.md`
13. `docs/architecture.md`
14. `docs/ROADMAP.md`
15. `docs/PUBLISH_CHECKLIST.md`
16. `docs/PYPI_PUBLISH.md` (when releasing Python bindings)
17. `docs/PYNABLED_GAPS_AUDIT.md` (when working on `feat/pynabled-bindings` / `pynabled`)
18. `docs/PYNABLED_PARITY_MATRIX.md` (when closing `N-PY-*` implementation gaps)
19. `docs/PYNABLED_ARCHITECTURE.md` (when changing Python carriers, copy contracts, or result objects)
20. `docs/STATUS.md`
21. `docs/EMBEDDINGS.md` (when working on retrieval compute / `nabled-embeddings` / `pynabled.embeddings`)
22. `docs/BENCHMARKS.md` (when benchmarking embeddings workloads)
23. `docs/NDARROW_INTEGRATION.md`
24. `docs/ARROW_SUPPORT_MATRIX.md`

Use `docs/EXECUTION_TRACKER.md` to resume from `Next` items first; avoid full code re-assessment unless tracker state is stale or contradictory.

Then verify repository state quickly:

1. `cargo metadata --no-deps`
2. `find crates -maxdepth 3 -type f | sort`
3. `find crates/nabled/src -maxdepth 2 -type f | sort`

## Context Sufficiency Check

After reading the docs above, a contributor should be able to answer:

1. What is the canonical compute substrate? (`ndarray`)
2. Is workspace migration complete? (Yes for library domains; check `docs/STATUS.md`)
3. Is current functionality sufficient for target scope? (Check `docs/CAPABILITY_MATRIX.md` verdict)
4. What is done vs next vs needed? (Check `docs/EXECUTION_TRACKER.md`)
5. Where does most implementation currently live? (`crates/nabled-linalg` and `crates/nabled-ml`)
6. What is the next milestone? (Benchmark-driven optimization and post-v1 expansion priorities; see `docs/EXECUTION_TRACKER.md`)
7. What quality gates are mandatory before finalizing changes? (`just checks`, clippy, tests, coverage floor in both internal/provider modes)
8. What is the backend feature contract? (`blas` baseline + provider feature policy; see `docs/DECISIONS.md`)
9. What are the execution axes? (`Provider` vs `Backend` vs `Kernel`; see `docs/DECISIONS.md`)
10. What is the complete kernel set and current wiring status? (See `docs/KERNEL_CATALOG.md`)
11. What objective rubric defines "done" and "production-ready v1"? (See `docs/REFERENCE_RUBRIC.md`)
12. What are the canonical Python carriers and copy/result contracts for `pynabled`? (See `docs/PYNABLED_ARCHITECTURE.md`)

## Scope Boundary

`nabled` core numerical crates remain Arrow-free and operate on ndarray data structures.

Optional Arrow-facing interop may exist only in facade-level feature-gated modules (currently the
`arrow` feature backed by `ndarrow`).
