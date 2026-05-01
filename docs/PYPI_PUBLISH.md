# Publishing pynabled to PyPI

This document describes how to release the **pynabled** Python package to PyPI. Rust crates (`nabled`, etc.) are published to crates.io separately; this flow only covers the Python bindings.

## Prerequisites

1. **PyPI Trusted Publisher**: Configure a PyPI Trusted Publisher for
   `MontOpsInc/nabled`, workflow `.github/workflows/publish-pypi.yml`, environment
   `pypi`, and project name `pynabled`.
   - If `pynabled` does not exist on PyPI yet, create a pending publisher from the
     account publishing settings. The pending publisher creates the project on first
     successful upload.
   - If the project already exists, add the publisher from the project's publishing
     settings.
2. **Optional TestPyPI Trusted Publisher**: Configure the same repository/workflow with
   environment `testpypi` on [TestPyPI](https://test.pypi.org) for rehearsals.
3. **GitHub environments**: Create repository environments named `pypi` and
   `testpypi`. Environment protection is optional but recommended for the real PyPI
   environment.

No long-lived `PYPI_API_TOKEN` or `TEST_PYPI_API_TOKEN` secret is required for the
normal release path.

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
python -m pip install "maturin==1.13.1"
maturin build --release --locked --out dist
```

Using `--out dist` matches CI and keeps wheels in one place (the default without `--out` is often `target/wheels/`). Fix any compile or packaging errors before tagging.
The repository tracks `Cargo.lock` intentionally so CI release wheels can build with
`--locked` from a clean checkout.

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

### 5. Optional: local metadata check

The publish workflow runs PyPA's upload action, which verifies package metadata before upload.
For a local approximation, inspect the wheel and sdist built above and run `just wheel-smoke` /
`just wheel-smoke-pytest` before tagging.

### 6. Optional: TestPyPI

Use the **workflow_dispatch** path on **Publish pynabled to PyPI** with
**use_testpypi** enabled. This runs the same multi-platform build matrix and publishes
the downloaded artifacts with OIDC Trusted Publishing to TestPyPI.

After the workflow succeeds:

```bash
pip install -i https://test.pypi.org/simple/ pynabled==<version>
```

## CI

The main **CI** workflow includes:

- **Python package version alignment** — `pyproject.toml` vs `Cargo.toml` workspace version.
- **Python wheel smoke** — `maturin build --release`, install wheel in a clean venv, import `pynabled` and `pynabled._pynabled`.
- **Python dependency audit** — the security workflow installs pinned audit tooling
  (`pip-audit==2.10.0`) and audits the resolved Python package/tooling requirement set when
  Python packaging inputs change.

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

That push triggers [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml), which builds **manylinux 2_28** (x86_64, aarch64), **Windows** (x64), **macOS** (x86_64 on `macos-13`, aarch64 on `macos-latest`), plus an **sdist**, then publishes the assembled artifacts to PyPI with PyPA Trusted Publishing.

Release automation is intentionally pinned:

1. `PyO3/maturin-action@v1.51.0`
2. `maturin v1.13.1`
3. `pypa/gh-action-pypi-publish@v1.14.0`

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
workflow. The default wheel now includes the Rust `arrow` feature. Provider/backend features
(`openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`, `magma-system`,
`accelerator-rayon`, and `accelerator-wgpu`) remain source-build workflows. Local/CI package gates
therefore smoke publish-style `wheel` / `sdist` artifacts for the default feature set, while
optional provider/backend permutations are validated as isolated source-build installs. There are
no Python extras that enable these Rust features; use the explicit feature names through the
`pynabled-provider`, `pynabled-accelerators`, and `pynabled-features` build settings (or raw
`maturin` build args for uncommon cases) as described in [BUILD.md](../BUILD.md).

## Troubleshooting

- **Invalid publisher / OIDC failure**: PyPI or TestPyPI publisher configuration must
  exactly match repository owner `MontOpsInc`, repository `nabled`, workflow
  `.github/workflows/publish-pypi.yml`, and the GitHub environment (`pypi` or
  `testpypi`).
- **Tag / version mismatch**: Tag must be exactly `pypi-v` + semver matching `Cargo.toml` and `pyproject.toml`.
- **File already exists**: PyPI does not allow re-uploading the same file version; bump the version and release again.
