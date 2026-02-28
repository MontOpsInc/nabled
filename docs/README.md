# Nabled Docs

This folder contains the current, authoritative direction for `nabled`.

## Current Direction

1. `nabled` is becoming an `ndarray`-first library.
2. Interop between Rust linalg libraries is not a current objective.
3. Performance, correctness, and composability are first-class requirements.
4. Workspace architecture is the foundation for scale.

## Documents

1. `docs/DECISIONS.md`: locked decisions and constraints.
2. `docs/architecture.md`: target package/module architecture.
3. `docs/ROADMAP.md`: implementation sequencing and milestones.
4. `docs/STATUS.md`: current migration snapshot (source of truth for what has been moved).

## Context Resume Protocol

When starting from a compacted/partial context, read documents in this order:

1. `docs/README.md`
2. `docs/DECISIONS.md`
3. `docs/architecture.md`
4. `docs/ROADMAP.md`
5. `docs/STATUS.md`

Then verify repository state quickly:

1. `cargo metadata --no-deps`
2. `find crates -maxdepth 3 -type f | sort`
3. `find crates/nabled/src -maxdepth 2 -type f | sort`

## Context Sufficiency Check

After reading the docs above, a contributor should be able to answer:

1. What is the canonical compute substrate? (`ndarray`)
2. Is workspace migration complete? (Yes for library domains; check `docs/STATUS.md`)
3. Where does most implementation currently live? (`crates/nabled-linalg` and `crates/nabled-ml`)
4. What is the next migration milestone? (Contract hardening and production-readiness work)
5. What quality gates are mandatory before finalizing changes? (`just checks`, clippy, tests, coverage floor)

## Scope Boundary

`nabled` has no Arrow knowledge. It operates on ndarray data structures.

Arrow-facing interop and execution belong in downstream libraries (for example, `narrow`).
