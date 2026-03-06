#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

PROBE_ITERS="${NABLED_GPU_PROBE_ITERS:-4}"
SIZES="${NABLED_GPU_PROBE_SIZES:-96 128 160 192 224 256 288 320 384 448 512 640 768 896 1024}"

mkdir -p /tmp/xdg-runtime
cat >/tmp/nvidia_icd_egl.json <<'JSON'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0",
    "api_version": "1.3.278"
  }
}
JSON

export XDG_RUNTIME_DIR=/tmp/xdg-runtime
export VK_ICD_FILENAMES=/tmp/nvidia_icd_egl.json
export WGPU_BACKEND=vulkan

for n in ${SIZES}; do
  echo "=== N=${n} (release) ==="
  out="$(
    NABLED_GPU_PROBE_SIZE="${n}" \
    NABLED_GPU_PROBE_ITERS="${PROBE_ITERS}" \
    cargo +stable test -p nabled --test gpu_perf_probe --release \
      --no-default-features --features accelerator-wgpu -- --ignored --nocapture 2>&1
  )"
  echo "${out}" | grep -E "gpu perf probe starting|cpu total=|warmup max_abs_diff" || true
done
