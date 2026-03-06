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
3. `docs/CAPABILITY_MATRIX.md`
4. `docs/GPU_V2_TRACKER.md`
5. `docs/REMOTE_GPU_WORKFLOW.md`
6. `docs/EXECUTION_TRACKER.md`
7. `docs/architecture.md`
8. `docs/ROADMAP.md`
9. `docs/STATUS.md`

Do not infer status from memory. Use:
1. `docs/STATUS.md` for current migration state.
2. `docs/CAPABILITY_MATRIX.md` for scope gaps and sufficiency.
3. `docs/EXECUTION_TRACKER.md` for `Done / Next / Needed` execution state.
4. Resume from highest-priority open `N-*` item in `docs/EXECUTION_TRACKER.md` unless explicitly redirected.

## Non-Negotiable Constraints

1. Canonical compute substrate is `ndarray`.
2. No `nalgebra`/`nalgebra-lapack` dependencies or code paths.
3. No hidden copy-heavy conversions in hot paths.
4. Public APIs stay pure numerical APIs over ndarray types.
5. `nabled` has no Arrow-aware API surface.
6. No legacy compatibility shims for unreleased APIs; prefer clean replacements over aliases/wrappers.
7. Use concise domain API naming (for example, decomposition modules expose `decompose`).
8. Performance-sensitive kernels should provide explicit allocation-control paths (`*_into`) and reusable workspace types when beneficial.
9. Keep execution-axis terminology explicit and consistent:
   - `Provider` = decomposition implementation source,
   - `Backend` = primitive-kernel execution target,
   - `Kernel` = operation-family backend contract.

## Workspace Migration Rules

1. Keep `crates/nabled` as a thin facade crate that re-exports workspace crates.
2. New algorithm implementations belong in `crates/nabled-linalg` or `crates/nabled-ml`, not `crates/nabled/src/`.
3. Keep behavior stable while relocating or reshaping modules:
   - preserve tests/examples/benches coverage for touched domains,
   - prefer removing outdated names and compatibility wrappers instead of preserving them.
4. Update `docs/STATUS.md` in the same change set when migration state changes.

## Quality Gates

Run and pass before finalizing:

1. `just checks` (preferred)
2. `cargo +stable clippy --workspace --no-default-features --features "openblas-system accelerator-rayon accelerator-wgpu" --all-targets -- -D warnings`
3. `cargo test --workspace --lib`
4. `cargo test -p nabled --tests`

Coverage expectation:

1. Keep line coverage > 90% as a hard gate for merge/push.
2. If coverage drops below this threshold, add meaningful tests before finalizing.
3. Prefer meaningful coverage over synthetic assertions.

## Test Placement

1. Unit tests in module-local `#[cfg(test)]` blocks.
2. `tests/` for cross-module integration/e2e behavior.

## Documentation Discipline

When architecture or behavior changes, update docs in the same change set:

1. `README.md` for user-visible behavior.
2. `docs/STATUS.md` for migration truth.
3. `docs/EXECUTION_TRACKER.md` for execution progression (`Done / Next / Needed`).
4. `docs/ROADMAP.md` if sequencing changes.
5. Relevant rustdoc comments for API contract changes.

## Remote GPU Workflow (Mandatory for CUDA/MAGMA Tasks)

For remote NVIDIA work, use the scripted tmux-first workflow from `docs/REMOTE_GPU_WORKFLOW.md`.

Defaults:

1. `SSH_USER=root`
2. `SSH_PORT=18800`
3. `SSH_KEY=~/.ssh/nabled_vast_4090`

Required flow:

1. Preferred single entrypoint:
   - `scripts/gpu_remote.sh up <host>`
   - `scripts/gpu_remote.sh one <host> magma-verify`
   - `scripts/gpu_remote.sh one <host> gpu-probe`
   - `scripts/gpu_remote.sh run <host> "<command>"`
   - `scripts/gpu_remote.sh attach <host>`
2. Equivalent low-level commands (if needed):
   - `scripts/gpu_remote_prepare.sh <host>`
   - `scripts/gpu_remote_tmux_session.sh <host>`
3. Launch standard job:
   - `scripts/gpu_remote.sh one <host> magma-verify`
   - `scripts/gpu_remote.sh one <host> gpu-probe`
4. Launch custom command:
   - `scripts/gpu_remote_tmux_run.sh <host> "<command>"`
5. Observe:
   - `scripts/gpu_remote_tmux_attach.sh <host>`

Do not run long remote jobs as ad hoc direct SSH commands when tmux scripts exist; observability and reproducibility are required.
Remote scripts intentionally ignore local SSH config (`-F /dev/null`) and pin explicit connection options to prevent host-specific drift.
Remote scripts use stdin-driven SSH execution for non-interactive steps to avoid provider command-mode wrappers.
