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
TMUX_SESSION="${TMUX_SESSION:-nabled-agent}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-20}"

SSH_OPTS=(
  -F /dev/null
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
  -o ControlMaster=no
  -o ControlPath=none
  -o ControlPersist=no
)

ssh -tt "${SSH_OPTS[@]}" -i "$SSH_KEY" -p "$SSH_PORT" "${SSH_USER}@${HOST}" <<EOF
tmux attach -t '${TMUX_SESSION}' || tmux list-sessions
EOF
