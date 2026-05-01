#!/usr/bin/env bash
# Fail if pyproject.toml [project].version != Cargo.toml [workspace.package].version
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CARGO_VERSION="$(awk '/^\[workspace\.package\]/ {in_ws=1; next} in_ws && /^version = / {gsub(/"/,"",$3); print $3; exit}' Cargo.toml)"
if [[ -z "$CARGO_VERSION" ]]; then
  echo "error: could not read [workspace.package] version from Cargo.toml"
  exit 1
fi

PY_VERSION="$(awk '/^\[project\]$/ {in_proj=1; next} in_proj && /^\[/ {in_proj=0} in_proj && /^version = / {gsub(/"/,"",$3); print $3; exit}' pyproject.toml)"
if [[ -z "$PY_VERSION" ]]; then
  echo "error: could not read version from pyproject.toml [project]"
  exit 1
fi

if [[ "$CARGO_VERSION" != "$PY_VERSION" ]]; then
  echo "error: version mismatch: Cargo.toml workspace.package = $CARGO_VERSION, pyproject.toml = $PY_VERSION"
  echo "Align versions before release (see docs/PYPI_PUBLISH.md)."
  exit 1
fi

echo "ok: pyproject.toml version matches Cargo.toml workspace.package ($CARGO_VERSION)"
