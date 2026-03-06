#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <host>"
  echo "example: SSH_USER=root SSH_PORT=18800 SSH_KEY=~/.ssh/nabled_vast_4090 $0 ssh9.vast.ai"
  exit 1
fi

HOST="$1"
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

ssh "${SSH_OPTS[@]}" -i "$SSH_KEY" -p "$SSH_PORT" "${SSH_USER}@${HOST}" <<EOF
set -euo pipefail
export HOME='${REMOTE_HOME}'
export PATH='${REMOTE_HOME}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
mkdir -p '${LOG_DIR}'
touch '${CURRENT_LOG}'

if tmux has-session -t '${TMUX_SESSION}' 2>/dev/null; then
  echo 'tmux session ${TMUX_SESSION} already exists'
  exit 0
fi

tmux new-session -d -s '${TMUX_SESSION}' -n work "bash --noprofile --norc -lc \"cd '${REMOTE_REPO_DIR}'; exec bash\""
tmux new-window -t '${TMUX_SESSION}' -n gpu 'watch -n 1 nvidia-smi'
tmux new-window -t '${TMUX_SESSION}' -n logs "bash --noprofile --norc -lc \"tail -n 200 -F '${CURRENT_LOG}'\""
tmux set-option -t '${TMUX_SESSION}' remain-on-exit on
tmux set-option -t '${TMUX_SESSION}' history-limit 200000
tmux set-option -t '${TMUX_SESSION}' mouse on
tmux pipe-pane -o -t '${TMUX_SESSION}:work.0' "cat >> '${LOG_DIR}/tmux-work-pane.log'"
echo 'created tmux session ${TMUX_SESSION}'
EOF

echo
echo "Session ready."
echo "Attach with:"
echo "  SSH_USER=${SSH_USER} SSH_PORT=${SSH_PORT} SSH_KEY=${SSH_KEY} TMUX_SESSION=${TMUX_SESSION} scripts/gpu_remote_tmux_attach.sh ${HOST}"
