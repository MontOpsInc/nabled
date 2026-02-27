# AGENTS.md

This file defines repository-specific rules for human and LLM contributors.

## Scope

Applies to the entire repository.

## Mission

Nabled is being built toward production readiness for numerical correctness, performance, and maintainability.

## Mandatory Quality Gates

Before proposing a final change set, run and pass:

1. `just checks`
2. `cargo +stable clippy --all-features --all-targets -- -D warnings` if `just checks` was not run.
3. Coverage validation when code paths changed in a meaningful way.

Minimum coverage expectation:

1. Keep total line coverage at or above 90%.
2. Prefer meaningful coverage over synthetic assertion-only tests.

## Test Placement Rules

1. Unit-level behavior belongs in module-local `#[cfg(test)]` blocks under `src/`.
2. `tests/` is reserved for cross-module integration/e2e style tests.
3. Do not add new catch-all integration test files for unit coverage when module-local tests are appropriate.

## CI Parity

CI enforces formatting, clippy `-D warnings`, unit tests, integration tests, and coverage upload.

Match CI locally by default:

1. `just checks`
2. `cargo llvm-cov --summary-only --all-features --all-targets`

## Architecture Direction (Current Priority)

Current top priority is backend/performance architecture.

1. Backend-first execution is preferred over adding broad new feature surface.
2. Design for easy backend expansion and zero-overhead abstractions whenever possible.
3. Avoid hidden conversions in hot paths.
4. Use static dispatch in performance-critical code paths unless there is a strong reason not to.

See [docs/PRODUCTION_READY.md](docs/PRODUCTION_READY.md) for the roadmap and design intent.

## Benchmark Expectations

Performance changes should include benchmark impact when relevant.

1. Prefer criterion benchmarks in `benches/`.
2. Benchmark against strong ecosystem baselines for the same operation where practical.
3. Do not claim performance wins without measured evidence.

## Safety Policy

1. Safe Rust by default.
2. `unsafe` is allowed only when justified by measurable performance or required interop.
3. Any `unsafe` must be narrowly scoped and documented with invariants.
4. Add tests that validate boundary conditions around unsafe code paths.

## Documentation Requirements

When behavior or architecture changes materially, update docs in the same change set:

1. `README.md` for user-visible behavior.
2. `docs/` plans/specs for architectural direction.
3. Inline rustdoc for API contract changes.

## Change Discipline

1. Keep changes focused and minimal for the task.
2. Do not silently alter unrelated workflows.
3. Preserve pedantic lint cleanliness.
