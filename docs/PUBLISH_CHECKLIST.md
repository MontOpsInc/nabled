# Publish Checklist

Last updated: 2026-03-03

## Purpose

This checklist is the publish gate for crates.io release readiness.

It is intentionally operational: each item should be either checked or blocked.

## Release Scope

Workspace crates:

1. `nabled-core`
2. `nabled-linalg`
3. `nabled-ml`
4. `nabled` (facade; publish last)

## Current Go/No-Go Status

Current state: **No-Go** until blockers are resolved.

Known blockers discovered from local packaging validation:

1. Internal workspace dependencies do not yet specify explicit version requirements for packaging.
   - Example failure from `cargo package -p nabled --allow-dirty`:
     - `dependency nabled-core does not specify a version`
2. `publish = false` is inherited from workspace package metadata and must be intentionally flipped when release is authorized.

## Pre-Publish Metadata Checklist

1. Ensure each publishable crate has:
   - `name`
   - `version`
   - `description`
   - `license`
   - `repository`
   - `documentation`
   - `readme`
   - `keywords` and `categories` where relevant.
2. Ensure internal cross-crate dependencies include explicit `version` requirements (path+version during workspace development is acceptable).
3. Ensure `publish` policy is set to allow publication for crates intended for crates.io.
4. Confirm crate-level README/docs phrasing is public-facing (not migration/internal wording).

## Quality Gate Checklist (Before Packaging)

1. `just checks`
2. `cargo +stable clippy --workspace --all-features --all-targets -- -D warnings`
3. `cargo test --workspace --lib`
4. `cargo test -p nabled --test integration`
5. Coverage line gate remains `> 90%`.

## Packaging Verification Checklist

Run in this order and fix all failures before proceeding:

1. `cargo package -p nabled-core`
2. `cargo check -p nabled-linalg`
3. `cargo check -p nabled-ml`
4. `cargo check -p nabled`
5. Optional leaf dry-run: `cargo publish --dry-run -p nabled-core --no-verify`

Important packaging nuance:

1. Before internal crates are published to crates.io for a new version, local `cargo package` for dependent crates may fail on crates.io index resolution (`nabled-linalg` expects published `nabled-core`, etc.).
2. Full dependent-crate packaging validation is therefore performed by ordered publish in CI release workflow (`core` -> `linalg` -> `ml` -> `nabled`).

Recommended local smoke install checks from package artifacts:

1. `cargo install --path crates/nabled-core --locked --debug`
2. `cargo install --path crates/nabled-linalg --locked --debug`
3. `cargo install --path crates/nabled-ml --locked --debug`
4. `cargo install --path crates/nabled --locked --debug`

## Publish Order Checklist

Publish in dependency order so crates.io index resolution is deterministic:

1. `cargo publish -p nabled-core`
2. `cargo publish -p nabled-linalg`
3. `cargo publish -p nabled-ml`
4. `cargo publish -p nabled`

If using CI workflow, ensure it mirrors this ordering; publishing `nabled` alone will fail if dependency versions are not available on crates.io.

## Post-Publish Verification

1. Confirm crates exist and versions resolve on crates.io for all four crates.
2. Confirm `cargo add nabled` resolves correct transitive versions.
3. Confirm docs.rs builds for `nabled` and links to `core/linalg/ml` APIs as expected.
4. Confirm README badges resolve for released version.
5. Tag release and publish release notes.

## Notes for Today

To publish today, highest-priority sequence is:

1. Add explicit internal dependency versions for packaging.
2. Switch publish policy from false to true for release-target crates.
3. Run packaging checks crate-by-crate.
4. Publish in dependency order (`core` -> `linalg` -> `ml` -> `nabled`).
