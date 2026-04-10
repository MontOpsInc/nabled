# Publishing pynabled to PyPI

This document describes how to release the **pynabled** Python package to PyPI. Rust crates (`nabled`, etc.) are published to crates.io separately; this flow only covers the Python bindings.

## Prerequisites

1. **PyPI project**: Register the `pynabled` name on [PyPI](https://pypi.org) if this is the first upload.
2. **GitHub secret**: Add `PYPI_API_TOKEN` in the repository **Settings → Secrets and variables → Actions**. Use a [PyPI API token](https://pypi.org/manage/account/token/) scoped to the project (or the whole account for the first upload).
3. **Optional — TestPyPI**: For a full upload rehearsal, create a token on [TestPyPI](https://test.pypi.org) and add `TEST_PYPI_API_TOKEN`. Use **Actions → Publish pynabled to PyPI → Run workflow** with **use_testpypi** enabled (this still runs the full multi-platform build matrix).

## Version policy

These must always match:

- `[workspace.package].version` in the root `Cargo.toml`
- `[project].version` in `pyproject.toml`

CI runs `scripts/check_pyproject_version_matches_cargo.sh` on every workflow. Fix any mismatch before merging or tagging.

## Pre-publish checks (beyond pytest)

Running `pytest` after `maturin develop` does **not** prove that release **wheels** install correctly.

### 1. Local: `maturin build --release`

From the repository root:

```bash
maturin build --release --out dist
```

Using `--out dist` matches CI and keeps wheels in one place (the default without `--out` is often `target/wheels/`). Fix any compile or packaging errors before tagging.

### 2. Local: clean venv + wheel install + import

```bash
python -m venv /tmp/pynabled-smoke
/tmp/pynabled-smoke/bin/pip install --upgrade pip
/tmp/pynabled-smoke/bin/pip install dist/pynabled-*.whl
/tmp/pynabled-smoke/bin/python -c "import pynabled; import pynabled._pynabled; print('ok')"
```

### 3. Local: one-liner via `just`

```bash
just wheel-smoke
```

### 4. Optional: full pytest against the wheel (slower)

```bash
just wheel-smoke-pytest
```

Installs the built wheel into a temporary venv and runs `python/tests/`.

### 5. Optional: `maturin publish --dry-run`

If your maturin version supports it, use `--dry-run` to exercise the publish path without uploading (behavior may vary by version; check `maturin publish --help`).

### 6. Optional: TestPyPI

Upload to TestPyPI and install from the test index:

```bash
# After building wheels locally or in CI artifacts
maturin publish --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ pynabled==<version>
```

Or use the **workflow_dispatch** path on **Publish pynabled to PyPI** with **use_testpypi** (requires `TEST_PYPI_API_TOKEN`).

## CI

The main **CI** workflow includes:

- **Python package version alignment** — `pyproject.toml` vs `Cargo.toml` workspace version.
- **Python wheel smoke** — `maturin build --release`, install wheel in a clean venv, import `pynabled` and `pynabled._pynabled`.

These run on code-changing PRs/pushes (wheel smoke is skipped for documentation-only changes).

## Release: tag-driven publish

Publishing is triggered by pushing a tag **`pypi-vX.Y.Z`** where **`X.Y.Z` equals** the workspace version in `Cargo.toml` and `pyproject.toml`.

### Recommended: `just tag-pypi-release`

From a **clean** git tree on the branch you want to release (usually `main`):

```bash
git checkout main
git pull
just tag-pypi-release
```

This will:

1. Set `pyproject.toml` `[project].version` from `Cargo.toml` `[workspace.package].version` (if needed).
2. Commit and push that change when the file was updated.
3. Create and push annotated tag `pypi-vX.Y.Z`.

That push triggers [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml), which builds **manylinux** (x86_64, aarch64), **Windows** (x64), **macOS** (x86_64 on `macos-13`, aarch64 on `macos-latest`), plus an **sdist**, then publishes to PyPI with `maturin publish` (using `PYPI_API_TOKEN`).

### Manual tagging

If versions are already aligned:

```bash
VERSION="$(awk '/^\[workspace\.package\]/ {in_ws=1; next} in_ws && /^version = / {gsub(/"/,"",$3); print $3; exit}' Cargo.toml)"
git tag -a "pypi-v${VERSION}" -m "PyPI publish pynabled ${VERSION}"
git push origin "pypi-v${VERSION}"
```

## Post-release

1. Confirm the workflow run succeeded on GitHub Actions.
2. Open the [PyPI project page](https://pypi.org/project/pynabled/) and check the description, classifiers, and file list.
3. Smoke test: `pip install pynabled==X.Y.Z` in a fresh virtual environment.

## Default wheel features

Wheels on PyPI use **default** Cargo features for `pynabled` unless you change the publish
workflow. That means no optional provider/backend/Arrow features are compiled into the published
wheel by default (`openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`,
`magma-system`, `accelerator-rayon`, `accelerator-wgpu`, and `arrow` are all source-build
workflows). Local/CI package gates therefore smoke publish-style `wheel` / `sdist` artifacts only
for the default feature set, while optional provider/backend/Arrow permutations are validated as
isolated source-build installs. There are no Python extras that enable these Rust features; use
the explicit Cargo feature names described in [BUILD.md](../BUILD.md).

## Troubleshooting

- **Authentication failed**: `PYPI_API_TOKEN` missing, expired, or wrong scope.
- **Tag / version mismatch**: Tag must be exactly `pypi-v` + semver matching `Cargo.toml` and `pyproject.toml`.
- **File already exists**: PyPI does not allow re-uploading the same file version; bump the version and release again.
