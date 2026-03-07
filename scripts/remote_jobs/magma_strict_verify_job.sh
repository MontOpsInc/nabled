#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

ARTIFACT_DIR="${ARTIFACT_DIR:-coverage/gpu-v2/magma}"
STRICT_LOG="${ARTIFACT_DIR}/strict-verification.log"

mkdir -p "${ARTIFACT_DIR}"

if ! command -v just >/dev/null 2>&1; then
  cargo +stable install just --locked
fi

# Strict MAGMA routing:
# 1) disable runtime fallback-on-provider-failure
# 2) force threshold-gated decomposition paths to attempt MAGMA
export NABLED_MAGMA_STRICT=1
export NABLED_MAGMA_MIN_DECOMPOSITION_DIM=1
export NABLED_MAGMA_BATCH_MIN_DECOMPOSITION_COUNT=1
export NABLED_MAGMA_BATCH_MIN_DECOMPOSITION_DIM=1
export NABLED_MAGMA_BATCH_MIN_DECOMPOSITION_WORK=1

{
  echo "# MAGMA Strict Verification"
  date -u
  echo
  echo "## Environment"
  nvidia-smi -L || true
  pkg-config --modversion magma || true
  echo
  echo "## Strict Routing Variables"
  env | grep '^NABLED_MAGMA_' | sort
  echo
  echo "## Build Validation"
  cargo +stable check --workspace --no-default-features --features magma-system
  cargo +stable clippy --workspace --no-default-features --features magma-system --all-targets -- -D warnings
  echo
  echo "## Strict MAGMA Test Matrix"
  cargo +stable test -p nabled-linalg --no-default-features --features magma-system --lib -- --nocapture --show-output
  cargo +stable test -p nabled --no-default-features --features magma-system --tests -- --nocapture --show-output
  echo
  echo "## Capability Report"
  cargo +stable run -p nabled --no-default-features --features magma-system --bin backend_capability_report \
    -- --output-dir coverage/backend-capabilities/magma-system-strict
} | tee "${STRICT_LOG}"

echo "magma strict verification artifacts are under ${ARTIFACT_DIR}"
