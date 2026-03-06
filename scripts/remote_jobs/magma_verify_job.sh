#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

RUN_BENCH_COMPARE="${RUN_BENCH_COMPARE:-1}"
BASELINE_PROVIDER="${BASELINE_PROVIDER:-openblas-system}"
ARTIFACT_DIR="${ARTIFACT_DIR:-coverage/gpu-v2/magma}"
BENCH_DIR="${ARTIFACT_DIR}/bench"
VERIFY_LOG="${ARTIFACT_DIR}/verification.log"

mkdir -p "${ARTIFACT_DIR}"

if ! command -v just >/dev/null 2>&1; then
  cargo +stable install just --locked
fi

{
  echo "# MAGMA Verification"
  date -u
  echo
  echo "## Environment"
  nvidia-smi -L || true
  pkg-config --modversion magma || true
  echo
  echo "## Build Validation"
  cargo +stable check --workspace --no-default-features --features magma-system
  cargo +stable clippy --workspace --no-default-features --features magma-system --all-targets -- -D warnings
  echo
  echo "## Correctness Validation"
  cargo +stable test --workspace --no-default-features --features magma-system --lib -- --nocapture --show-output
  cargo +stable test -p nabled --no-default-features --features magma-system --tests -- --nocapture --show-output
  echo
  echo "## Capability Report"
  cargo +stable run -p nabled --no-default-features --features magma-system --bin backend_capability_report \
    -- --output-dir coverage/backend-capabilities/magma-system
} | tee "${VERIFY_LOG}"

if [[ "${RUN_BENCH_COMPARE}" == "1" ]]; then
  mkdir -p "${BENCH_DIR}"

  if [[ -n "${BASELINE_PROVIDER}" ]]; then
    NABLED_PROVIDER_BENCH_FEATURES="${BASELINE_PROVIDER}" just -f .justfile bench-smoke-report-provider
    cp coverage/benchmarks/summary.json "${BENCH_DIR}/${BASELINE_PROVIDER}-summary.json"
  fi

  NABLED_PROVIDER_BENCH_FEATURES="magma-system" just -f .justfile bench-smoke-report-provider
  cp coverage/benchmarks/summary.json "${BENCH_DIR}/magma-system-summary.json"
fi

echo "magma verification artifacts are under ${ARTIFACT_DIR}"
