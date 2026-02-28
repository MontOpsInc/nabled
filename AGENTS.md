# AGENTS.md

Repository-level instructions for human and LLM contributors.

## Scope

Applies to the entire repository.

## Mission

Build `nabled` into a production-grade, ndarray-native numerical library with strong correctness, performance, and maintainability.

## Mandatory Context Bootstrap

Before making architectural or broad refactor changes, read these in order:

1. `docs/README.md`
2. `docs/DECISIONS.md`
3. `docs/architecture.md`
4. `docs/ROADMAP.md`
5. `docs/STATUS.md`

Do not infer status from memory. Use `docs/STATUS.md` as the source of truth.

## Non-Negotiable Constraints

1. Canonical compute substrate is `ndarray`.
2. No `nalgebra`/`nalgebra-lapack` dependencies or code paths.
3. No hidden copy-heavy conversions in hot paths.
4. Public APIs stay pure numerical APIs over ndarray types.
5. `nabled` has no Arrow-aware API surface.

## Workspace Migration Rules

1. Keep `crates/nabled` as a thin facade crate that re-exports workspace crates.
2. New algorithm implementations belong in `crates/nabled-linalg` or `crates/nabled-ml`, not `crates/nabled/src/`.
3. Keep behavior stable while relocating or reshaping modules:
   - preserve tests/examples/benches coverage for touched domains,
   - keep public API compatibility unless an explicit change is requested.
4. Update `docs/STATUS.md` in the same change set when migration state changes.

## Quality Gates

Run and pass before finalizing:

1. `just checks` (preferred)
2. `cargo +stable clippy --workspace --all-features --all-targets -- -D warnings`
3. `cargo test --workspace --lib`
4. `cargo test -p nabled --test integration`

Coverage expectation:

1. Keep line coverage >= 90%.
2. Prefer meaningful coverage over synthetic assertions.

## Test Placement

1. Unit tests in module-local `#[cfg(test)]` blocks.
2. `tests/` for cross-module integration/e2e behavior.

## Documentation Discipline

When architecture or behavior changes, update docs in the same change set:

1. `README.md` for user-visible behavior.
2. `docs/STATUS.md` for migration truth.
3. `docs/ROADMAP.md` if sequencing changes.
4. Relevant rustdoc comments for API contract changes.
