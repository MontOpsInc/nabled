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
PROBE_ITERS="${NABLED_GPU_PROBE_ITERS:-4}"
SIZES="${NABLED_GPU_PROBE_SIZES:-96 128 160 192 224 256 288 320 384 448 512 640 768 896 1024}"
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
cd \"${REMOTE_REPO_DIR}\"

mkdir -p /tmp/xdg-runtime
cat >/tmp/nvidia_icd_egl.json <<\"JSON\"
{
  \"file_format_version\": \"1.0.0\",
  \"ICD\": {
    \"library_path\": \"libEGL_nvidia.so.0\",
    \"api_version\": \"1.3.278\"
  }
}
JSON

export XDG_RUNTIME_DIR=/tmp/xdg-runtime
export VK_ICD_FILENAMES=/tmp/nvidia_icd_egl.json
export WGPU_BACKEND=vulkan

for n in ${SIZES}; do
  echo \"=== N=${n} (release) ===\"
  out=\$(NABLED_GPU_PROBE_SIZE=\"${n}\" NABLED_GPU_PROBE_ITERS=\"${PROBE_ITERS}\" \
    cargo +stable test -p nabled --test gpu_perf_probe --release \
      --no-default-features --features accelerator-wgpu -- --ignored --nocapture 2>&1)
  echo \"\${out}\" | rg \"gpu perf probe starting|cpu total=|warmup max_abs_diff\" || true
done
'"
