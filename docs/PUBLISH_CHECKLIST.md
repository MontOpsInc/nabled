# Publish Checklist

Last updated: 2026-05-28

This document is the release-day checklist for publishing the `nabled` workspace crates.

## Scope

Publish order is fixed:

1. `nabled-core`
2. `nabled-linalg`
3. `nabled-ml`
4. `nabled-embeddings`
5. `nabled-kinematics`
6. `nabled-model`
7. `nabled-dynamics`
8. `nabled-control`
9. `nabled-sensor`
10. `nabled-sim`
11. `nabled` (facade)

**Python package (pynabled on PyPI):** independent of the crates.io order above. See [PYPI_PUBLISH.md](PYPI_PUBLISH.md) for tags (`pypi-v*`), Trusted Publishing setup, wheel smoke tests, and `just tag-pypi-release`.

## crates.io rate limits

Reference: [crates.io rate limits](https://crates.io/docs/rate-limits).

| Publish type | Burst (at once) | Sustained (after burst) |
|--------------|-----------------|-------------------------|
| **Brand-new crate name** | **5** per account | **1 every 10 minutes** |
| **New version of existing crate** | **30** per account | **1 per minute** |

On exceed: HTTP **429 Too Many Requests** with a “try again after … GMT” timestamp.

**Workspace rules:**

1. Never register **six or more new `nabled-*` crate names** in a single Release workflow run (0.0.9 lesson).
2. `.github/workflows/release.yml` paces version bumps with **65s** sleeps and new crates with **65s** (first five) then **600s** after the fifth new name.
3. On 429: wait until the crates.io timestamp, then re-run Release via `workflow_dispatch` with skip flags for crates already published.

## Co-ownership policy

Both maintainers must appear on each crate's **Owners** list on crates.io.

When a maintainer needs access to crates registered by the other:

1. Merge `.github/workflows/crates-io-add-owner.yml` to `main`.
2. Run the **crates.io add owner** workflow (`workflow_dispatch`) with the target username.
3. Uses existing org/repo `CARGO_REGISTRY_TOKEN` — **no secret rotation required**.
4. Verify Owners on crates.io (George + NiklausParcell).

Alternative: an existing owner runs `cargo owner --add <username> -p <crate>` locally.

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
3. Updates README dependency snippets (core, linalg, ml, and Physical AI crates).
4. Updates lockfile metadata.
5. Generates `CHANGELOG.md` and `RELEASE_NOTES.md`.
6. Commits and pushes the release branch.

Then:

1. Open PR `release-vX.Y.Z` -> `main`.
2. Verify CI is green.
3. Merge PR.

## First-time co-owner add (before tag, when needed)

After workflow and doc changes merge to `main`:

1. Actions → **crates.io add owner** → Run workflow (default: `NiklausParcell`).
2. Confirm both maintainers on Owners for all workspace crates.
3. Proceed to tag (below).

## Tag and Publish

After merge:

1. `just tag-release X.Y.Z`

This creates/pushes `vX.Y.Z` and triggers `.github/workflows/release.yml`, which publishes crates in dependency order (~12–15 minutes for version-bump releases).

### Partial publish recovery

If Release fails mid-run:

1. Note which crates reached `X.Y.Z` on crates.io.
2. Re-run Release via **workflow_dispatch** on the same tag commit.
3. Use skip inputs for crates already at `X.Y.Z` (`skip_core`, `skip_kinematics`, etc.).
4. The workflow skips versions that already exist and retries 429 after waiting.

Then for PyPI:

1. `just tag-pypi-release` (creates `pypi-vX.Y.Z` from workspace version)

## Release Workflow Expectations

`release.yml` must continue to guarantee:

1. Version consistency check against `[workspace.package]`.
2. Ordered publish (core → linalg → ml → Physical AI → facade).
3. Rate-limit pacing (65s version bumps; 600s after fifth new crate name).
4. Resume on “version already exists” and 429 retry.
5. Optional dry-run verification via workflow input.
6. GitHub Release creation in a separate job (only when all publishes succeed).

## Common Failure Modes

1. **429 Too Many Requests** on crates.io:
   - wait for the timestamp in the error message;
   - re-run workflow with skip flags for completed crates.
2. **Six new crate names in one run:**
   - split across multiple workflow runs or days; never loop six new names in one job.
3. **`netlib-system` link errors on macOS** (`ld: library 'gfortran' not found`):
   - install toolchain/runtime (`brew install gcc`), or
   - keep local validation at compile/lint level and rely on CI publish path.
4. **Static provider build failures:**
   - ensure `gcc`, `gfortran`, `make` are installed.
5. **Dirty working tree before `prepare-release`:**
   - commit/stash first.
6. **Missing co-owner on a crate:**
   - run `crates-io-add-owner` workflow or `cargo owner --add`.
