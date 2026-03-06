#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <host>"
  echo "example: SSH_PORT=18800 SSH_KEY=~/.ssh/nabled_vast_4090 $0 ssh9.vast.ai"
  exit 1
fi

HOST="$1"
SSH_USER="${SSH_USER:-nabled}"
SSH_PORT="${SSH_PORT:-18800}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/nabled_vast_4090}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/${SSH_USER}/nabled}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-20}"
RUN_BENCH_COMPARE="${RUN_BENCH_COMPARE:-1}"
BASELINE_PROVIDER="${BASELINE_PROVIDER:-openblas-system}"

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
  -o ControlMaster=no
  -o ControlPath=none
  -o ControlPersist=no
)

ssh "${SSH_OPTS[@]}" -i "$SSH_KEY" -p "$SSH_PORT" "${SSH_USER}@${HOST}" "bash --noprofile --norc -c '
set -euo pipefail
export HOME=/home/${SSH_USER}
export PATH=/home/${SSH_USER}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd \"${REMOTE_REPO_DIR}\"

mkdir -p coverage/gpu-v2/magma

{
  echo \"# MAGMA Verification\"
  date -u
  echo
  echo \"## Environment\"
  nvidia-smi -L || true
  pkg-config --modversion magma || true
  echo
  echo \"## Build Validation\"
  cargo +stable check --workspace --no-default-features --features magma-system
  cargo +stable clippy --workspace --no-default-features --features magma-system --all-targets -- -D warnings
  echo
  echo \"## Correctness Validation\"
  cargo +stable test --workspace --no-default-features --features magma-system --lib -- --nocapture --show-output
  cargo +stable test -p nabled --no-default-features --features magma-system --tests -- --nocapture --show-output
  cargo +stable run -p nabled --no-default-features --features magma-system --bin backend_capability_report \
    -- --output-dir coverage/backend-capabilities/magma-system
} | tee coverage/gpu-v2/magma/verification.log

if [[ \"${RUN_BENCH_COMPARE}\" == \"1\" ]]; then
  mkdir -p coverage/gpu-v2/magma/bench

  if [[ -n \"${BASELINE_PROVIDER}\" ]]; then
    NABLED_PROVIDER_BENCH_FEATURES=\"${BASELINE_PROVIDER}\" just -f .justfile bench-smoke-report-provider
    cp coverage/benchmarks/summary.json \"coverage/gpu-v2/magma/bench/${BASELINE_PROVIDER}-summary.json\"
  fi

  NABLED_PROVIDER_BENCH_FEATURES=magma-system just -f .justfile bench-smoke-report-provider
  cp coverage/benchmarks/summary.json coverage/gpu-v2/magma/bench/magma-system-summary.json
fi

echo \"magma verification artifacts are under coverage/gpu-v2/magma\"
'"
