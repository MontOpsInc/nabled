#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <host>"
  echo "example: SSH_USER=root SSH_PORT=18800 SSH_KEY=~/.ssh/nabled_vast_4090 $0 ssh9.vast.ai"
  echo "set NABLED_REMOTE_FORCE_BOOTSTRAP=1 to force apt/rust/bootstrap even on pre-baked images"
  exit 1
fi

HOST="$1"
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-18800}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/nabled_vast_4090}"
REPO_URL="${REPO_URL:-https://github.com/MontOpsInc/nabled.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REMOTE_HOME="${REMOTE_HOME:-$([[ "${SSH_USER}" == "root" ]] && echo "/root" || echo "/home/${SSH_USER}")}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-${REMOTE_HOME}/nabled}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-20}"
NABLED_REMOTE_FORCE_BOOTSTRAP="${NABLED_REMOTE_FORCE_BOOTSTRAP:-0}"
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

SUDO=sudo
if [[ \$(id -u) -eq 0 ]]; then
  SUDO=
elif ! command -v sudo >/dev/null 2>&1; then
  SUDO=
fi

# Vast injects root SSH keys by default; mirror to agent for optional agent-mode sessions.
if [[ '${SSH_USER}' == 'root' && -f /root/.ssh/authorized_keys && -d /home/agent ]]; then
  install -d -m 700 -o agent -g agent /home/agent/.ssh
  cp /root/.ssh/authorized_keys /home/agent/.ssh/authorized_keys
  chown agent:agent /home/agent/.ssh/authorized_keys
  chmod 600 /home/agent/.ssh/authorized_keys
fi

is_prebaked=0
if [[ -f /etc/nabled/nvidia-image ]]; then
  is_prebaked=1
fi

if [[ '${NABLED_REMOTE_FORCE_BOOTSTRAP}' == '1' || "\${is_prebaked}" == '0' ]]; then
  \${SUDO} apt-get update
  \${SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl ca-certificates build-essential pkg-config cmake clang libssl-dev gfortran \
    tmux vulkan-tools libmagma-dev libopenblas-dev liblapack-dev \
    ripgrep fd-find jq neovim \
    python3 python3-dev python3-venv python3-pip python3-setuptools python3-wheel \
    iputils-ping mtr-tiny iproute2 net-tools dnsutils traceroute netcat-openbsd

  if ! command -v rustup >/dev/null 2>&1; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y
  fi
  rustup toolchain install stable
  rustup default stable

  if ! command -v just >/dev/null 2>&1; then
    cargo +stable install just --locked
  fi

  if ! command -v maturin >/dev/null 2>&1; then
    python3 -m pip install --user --upgrade pip setuptools wheel maturin
  fi
else
  echo "detected pre-baked nabled NVIDIA image; skipping package bootstrap"
fi

if [[ -d '${REMOTE_REPO_DIR}/.git' ]]; then
  cd '${REMOTE_REPO_DIR}'
  git fetch origin
  git checkout '${REPO_BRANCH}'
  git pull --ff-only origin '${REPO_BRANCH}'
else
  git clone --branch '${REPO_BRANCH}' '${REPO_URL}' '${REMOTE_REPO_DIR}'
fi

cd '${REMOTE_REPO_DIR}'
mkdir -p /tmp/xdg-runtime

echo "remote gpu prep complete"
EOF
