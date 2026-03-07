#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

ARTIFACT_DIR="${ARTIFACT_DIR:-coverage/gpu-v2/magma}"
OUT_FILE="${ARTIFACT_DIR}/capability-scan.log"
mkdir -p "${ARTIFACT_DIR}"

MAGMA_LIB="${MAGMA_LIB:-$(ldconfig -p 2>/dev/null | awk '/libmagma\.so( |$)/ {print $NF; exit}')}"
MAGMA_SPARSE_LIB="${MAGMA_SPARSE_LIB:-$(ldconfig -p 2>/dev/null | awk '/libmagma_sparse\.so( |$)/ {print $NF; exit}')}"

{
  echo "# MAGMA Capability Scan"
  date -u
  echo
  echo "## GPU"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  echo "CUDA_TOOLKIT: $(nvcc --version 2>/dev/null | rg -o 'release [0-9.]+' -m1 || echo unknown)"
  echo
  echo "## Libraries"
  echo "magma: ${MAGMA_LIB:-missing}"
  echo "magma_sparse: ${MAGMA_SPARSE_LIB:-missing}"
  echo
  echo "## Sparse Symbols"
  if [[ -n "${MAGMA_SPARSE_LIB}" ]]; then
    sparse_count="$(nm -D "${MAGMA_SPARSE_LIB}" 2>/dev/null | rg -c 'spmv|spmm|cg|gmres|jacobi|ilu|ic|csr|csc|parilut' || true)"
    echo "count=${sparse_count}"
    nm -D "${MAGMA_SPARSE_LIB}" 2>/dev/null \
      | rg 'spmv|spmm|cg|gmres|jacobi|ilu|ic|csr|csc|parilut' -N \
      | head -80 || true
  else
    echo "count=0"
  fi
  echo
  echo "## Mixed-Precision/Refinement Symbols"
  if [[ -n "${MAGMA_LIB}" ]]; then
    mixed_count="$(nm -D "${MAGMA_LIB}" 2>/dev/null | rg -c 'dsgesv|zcgesv|iteref|gesv_rbt|gesv_gpu' || true)"
    echo "count=${mixed_count}"
    nm -D "${MAGMA_LIB}" 2>/dev/null \
      | rg 'dsgesv|zcgesv|iteref|gesv_rbt|gesv_gpu' -N \
      | head -80 || true
  else
    echo "count=0"
  fi
  echo
  echo "## Batched Decomposition Symbols"
  if [[ -n "${MAGMA_LIB}" ]]; then
    echo "### Present and used"
    used_count="$(nm -D "${MAGMA_LIB}" 2>/dev/null | rg -c 'getrf_batched|potrf_batched|geqrf_batched' || true)"
    echo "count=${used_count}"
    nm -D "${MAGMA_LIB}" 2>/dev/null \
      | rg 'getrf_batched|potrf_batched|geqrf_batched' -N \
      | head -80 || true
    echo
    echo "### Desired (SVD / symmetric eigen)"
    desired_count="$(
      nm -D "${MAGMA_LIB}" 2>/dev/null \
        | rg -c 'gesvd(_batched)?|gesdd(_batched)?|gesvdj(_batched)?|gesvdx(_batched)?|syevd(_batched)?|heevd(_batched)?' || true
    )"
    echo "count=${desired_count}"
    nm -D "${MAGMA_LIB}" 2>/dev/null \
      | rg 'gesvd(_batched)?|gesdd(_batched)?|gesvdj(_batched)?|gesvdx(_batched)?|syevd(_batched)?|heevd(_batched)?' -N \
      | head -120 || true
  else
    echo "count=0"
  fi
  echo
  echo "## Headers"
  ls /usr/include/magma*.h /usr/include/magmasparse*.h 2>/dev/null | sort || true
} | tee "${OUT_FILE}"

echo "capability scan written: ${OUT_FILE}"
