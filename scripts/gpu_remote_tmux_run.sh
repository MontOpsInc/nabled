#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <host> <command...>"
  echo "example: SSH_USER=root $0 ssh9.vast.ai 'bash scripts/remote_jobs/magma_verify_job.sh'"
  exit 1
fi

HOST="$1"
shift
USER_COMMAND="$*"

SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-18800}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/nabled_vast_4090}"
REMOTE_HOME="${REMOTE_HOME:-$([[ "${SSH_USER}" == "root" ]] && echo "/root" || echo "/home/${SSH_USER}")}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-${REMOTE_HOME}/nabled}"
TMUX_SESSION="${TMUX_SESSION:-nabled-agent}"
LOG_DIR="${LOG_DIR:-${REMOTE_HOME}/.cache/nabled-agent/logs}"
CURRENT_LOG="${CURRENT_LOG:-${LOG_DIR}/current.log}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-20}"

SSH_OPTS=(
  -T
  -F /dev/null
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
  -o ControlMaster=no
  -o ControlPath=none
  -o ControlPersist=no
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/gpu_remote_tmux_session.sh" "${HOST}"

COMMAND_B64="$(printf '%s' "${USER_COMMAND}" | base64 | tr -d '\n')"

ssh "${SSH_OPTS[@]}" -i "$SSH_KEY" -p "$SSH_PORT" "${SSH_USER}@${HOST}" <<EOF
set -euo pipefail
export HOME='${REMOTE_HOME}'
export PATH='${REMOTE_HOME}/.cargo/bin:/home/agent/.cargo/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'

if ! tmux has-session -t '${TMUX_SESSION}' 2>/dev/null; then
  echo 'tmux session ${TMUX_SESSION} does not exist' >&2
  exit 1
fi

mkdir -p '${LOG_DIR}'
ts=\$(date -u +%Y%m%dT%H%M%SZ)
job_script='${LOG_DIR}/job-'\${ts}'.sh'
job_log='${LOG_DIR}/job-'\${ts}'.log'
runner_script='${LOG_DIR}/runner-'\${ts}'.sh'
printf '%s' '${COMMAND_B64}' | base64 --decode >"\${job_script}"
chmod +x "\${job_script}"
ln -sfn "\${job_log}" '${CURRENT_LOG}'

cat >"\${runner_script}" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail
export HOME='__REMOTE_HOME__'
export PATH='__RUNNER_PATH__'
job_script='__JOB_SCRIPT__'
job_log='__JOB_LOG__'
cd '__REMOTE_REPO_DIR__'
echo "[START] \$(date -u)" | tee -a "\${job_log}"
set +e
bash "\${job_script}" 2>&1 | tee -a "\${job_log}"
rc=\${PIPESTATUS[0]}
set -e
echo "[END rc=\${rc}] \$(date -u)" | tee -a "\${job_log}"
exit \${rc}
RUNNER
sed -i "s|__REMOTE_REPO_DIR__|${REMOTE_REPO_DIR}|g" "\${runner_script}"
sed -i "s|__REMOTE_HOME__|${REMOTE_HOME}|g" "\${runner_script}"
sed -i "s|__RUNNER_PATH__|${REMOTE_HOME}/.cargo/bin:/home/agent/.cargo/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin|g" "\${runner_script}"
sed -i "s|__JOB_SCRIPT__|\${job_script}|g" "\${runner_script}"
sed -i "s|__JOB_LOG__|\${job_log}|g" "\${runner_script}"
chmod +x "\${runner_script}"

tmux send-keys -t '${TMUX_SESSION}:work.0' C-c
tmux send-keys -t '${TMUX_SESSION}:work.0' "bash \${runner_script}" C-m

echo "launched job script: \${job_script}"
echo "launched runner script: \${runner_script}"
echo "logging to: \${job_log}"
EOF

echo
echo "Job submitted."
echo "Attach with:"
echo "  SSH_USER=${SSH_USER} SSH_PORT=${SSH_PORT} SSH_KEY=${SSH_KEY} TMUX_SESSION=${TMUX_SESSION} scripts/gpu_remote_tmux_attach.sh ${HOST}"
