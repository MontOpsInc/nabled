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
REPO_URL="${REPO_URL:-https://github.com/MontOpsInc/nabled.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/${SSH_USER}/nabled}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-20}"
SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)

ssh "${SSH_OPTS[@]}" -i "$SSH_KEY" -p "$SSH_PORT" "${SSH_USER}@${HOST}" "bash -lc '
set -euo pipefail
export HOME=/home/${SSH_USER}
export PATH=/home/${SSH_USER}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl ca-certificates build-essential pkg-config cmake clang libssl-dev \
  tmux vulkan-tools libmagma-dev

if ! command -v rustup >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
rustup toolchain install stable
rustup default stable

if [[ -d \"${REMOTE_REPO_DIR}/.git\" ]]; then
  cd \"${REMOTE_REPO_DIR}\"
  git fetch origin
  git checkout \"${REPO_BRANCH}\"
  git pull --ff-only origin \"${REPO_BRANCH}\"
else
  git clone --branch \"${REPO_BRANCH}\" \"${REPO_URL}\" \"${REMOTE_REPO_DIR}\"
fi

cd \"${REMOTE_REPO_DIR}\"
mkdir -p /tmp/xdg-runtime

echo \"remote gpu prep complete\"
'"
