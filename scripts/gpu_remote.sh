#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
usage:
  scripts/gpu_remote.sh <command> <host> [args...]

commands:
  prepare <host>                Prepare host and sync repository
  session <host>                Ensure tmux session exists
  up <host>                     Prepare + ensure tmux session
  one <host> [job]              Run named tmux job (magma-verify|magma-capability|gpu-probe|checks)
  run <host> <command...>       Run arbitrary command in tmux work pane
  attach <host>                 Attach to tmux session
  probe <host>                  Alias for: one <host> gpu-probe
  magma-verify <host>           Alias for: one <host> magma-verify
  magma-capability <host>       Alias for: one <host> magma-capability
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

COMMAND="$1"
HOST="$2"
shift 2

run_named_job() {
  local job="$1"
  local command=""
  case "${job}" in
    magma-verify)
      command="bash scripts/remote_jobs/magma_verify_job.sh"
      ;;
    magma-capability)
      command="bash scripts/remote_jobs/magma_capability_job.sh"
      ;;
    gpu-probe)
      command="bash scripts/remote_jobs/gpu_probe_job.sh"
      ;;
    checks)
      command="just checks"
      ;;
    *)
      echo "unknown job: ${job}" >&2
      echo "supported jobs: magma-verify | magma-capability | gpu-probe | checks" >&2
      exit 1
      ;;
  esac

  "${SCRIPT_DIR}/gpu_remote_prepare.sh" "${HOST}"
  "${SCRIPT_DIR}/gpu_remote_tmux_run.sh" "${HOST}" "${command}"
}

case "${COMMAND}" in
  prepare)
    "${SCRIPT_DIR}/gpu_remote_prepare.sh" "${HOST}"
    ;;
  session)
    "${SCRIPT_DIR}/gpu_remote_tmux_session.sh" "${HOST}"
    ;;
  up)
    "${SCRIPT_DIR}/gpu_remote_prepare.sh" "${HOST}"
    "${SCRIPT_DIR}/gpu_remote_tmux_session.sh" "${HOST}"
    ;;
  one)
    JOB="${1:-magma-verify}"
    run_named_job "${JOB}"
    ;;
  run)
    if [[ $# -lt 1 ]]; then
      echo "run requires a command payload" >&2
      usage
      exit 1
    fi
    "${SCRIPT_DIR}/gpu_remote_tmux_run.sh" "${HOST}" "$*"
    ;;
  attach)
    "${SCRIPT_DIR}/gpu_remote_tmux_attach.sh" "${HOST}"
    ;;
  probe)
    run_named_job "gpu-probe"
    ;;
  magma-verify)
    run_named_job "magma-verify"
    ;;
  magma-capability)
    run_named_job "magma-capability"
    ;;
  *)
    echo "unknown command: ${COMMAND}" >&2
    usage
    exit 1
    ;;
esac
