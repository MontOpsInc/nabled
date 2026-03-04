# Publish Checklist

Last updated: 2026-03-04

This document is the release-day checklist for publishing the `nabled` workspace crates.

## Scope

Publish order is fixed:

1. `nabled-core`
2. `nabled-linalg`
3. `nabled-ml`
4. `nabled` (facade)

## Feature Contract (Release-Relevant)

Supported provider features:

1. `openblas-system`
2. `openblas-static`
3. `netlib-system`
4. `netlib-static`

Execution notes:

1. Provider selection is compile-time.
2. `openblas-system` is the baseline provider path for local/CI test runs.
3. `netlib-system` is compile/lint validated in CI and local checks.
4. Static providers require toolchains (`gcc`, `gfortran`, `make`) and are validated in targeted environments.

## Pre-Release Gates

Run before creating a release branch:

1. `just checks`
2. `just check-provider-netlib`
3. `cargo package --allow-dirty -p nabled-core`

Optional (when static provider toolchains are installed):

1. `just check-provider-static`

Notes:

1. `nabled-core` is the leaf crate and can always be packaged locally.
2. Dependent crates are validated by ordered publish in the release workflow (after upstream crates are published).

## Versioning and Branch Prep

Use:

1. `just prepare-release X.Y.Z`

This performs:

1. Creates `release-vX.Y.Z` branch.
2. Updates `[workspace.package]` version and internal workspace dependency pins.
3. Updates README dependency snippets.
4. Updates lockfile metadata.
5. Generates `CHANGELOG.md` and `RELEASE_NOTES.md`.
6. Commits and pushes the release branch.

Then:

1. Open PR `release-vX.Y.Z` -> `main`.
2. Verify CI is green.
3. Merge PR.

## Tag and Publish

After merge:

1. `just tag-release X.Y.Z`

This creates/pushes `vX.Y.Z` and triggers `.github/workflows/release.yml`, which publishes crates in dependency order.

## Release Workflow Expectations

`release.yml` must continue to guarantee:

1. Version consistency check against `[workspace.package]`.
2. Ordered publish (`core` -> `linalg` -> `ml` -> facade).
3. Optional dry-run verification via workflow input.
4. GitHub Release creation from `RELEASE_NOTES.md`.

## Common Failure Modes

1. `netlib-system` link errors on macOS (`ld: library 'gfortran' not found`):
   - install toolchain/runtime (`brew install gcc`), or
   - keep local validation at compile/lint level and rely on CI publish path.
2. Static provider build failures:
   - ensure `gcc`, `gfortran`, `make` are installed.
3. Dirty working tree before `prepare-release`:
   - commit/stash first.
