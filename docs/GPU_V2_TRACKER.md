# GPU V2 Tracker

Last updated: 2026-03-06 (MAGMA remote verification and provider benchmark capture complete)

## Purpose

Track the v2 direction for GPU/provider expansion without losing v1 stabilization history.

This tracker is focused on:

1. Batch-oriented GPU workload coverage.
2. Runtime GPU routing policy for small vs large workloads.
3. MAGMA provider integration (CUDA first).
4. Verification harness for rapid remote iteration.

## Scope Lock

1. NVIDIA path first (CUDA + MAGMA-CUDA).
2. AMD/HIP path is explicitly deferred until hardware is available.
3. `Provider` and `Backend` remain orthogonal:
   - `Provider` (decomposition source) -> MAGMA/LAPACK/internal.
   - `Backend` (kernel target) -> CPU/GPU kernel dispatch.

## Work Items

| ID | Item | Status | Notes |
|---|---|---|---|
| `V2-001` | Remote GPU verification environment | Completed | Vast 4090 runbook is scripted (`scripts/gpu_v2_remote_prepare.sh`, `scripts/gpu_v2_remote_probe.sh`) and validated via tmux-driven remote execution. |
| `V2-002` | Batched surface audit and GPU relevance map | Completed | Decomposition batch APIs are provider-accelerated per-slice loops; kernel-level batch APIs are backend-dispatched and GPU-capable (`batched_matmat`, `batched_row_matvec`). |
| `V2-003` | Runtime workload policy (small vs large) for GPU backend routing | Completed | Centralized `accelerator::policy` now gates all GPU backend-dispatched kernels with env-overridable thresholds tuned from remote 4090 release-profile measurements. |
| `V2-004` | MAGMA CUDA provider integration (scaffold + domain wiring) | Completed | MAGMA provider paths are wired for LU/Cholesky/QR/SVD/symmetric eigen, with provider-safe scalar bounds propagated across dependent domains. |
| `V2-005` | MAGMA verification matrix + benchmark parity report | Completed | Remote RTX 4090 run completed with correctness/capability artifacts and provider benchmark summaries captured locally. |

## Batched Surface Snapshot (Current)

### Decomposition-level batched APIs (`nabled-linalg::batched`)

1. `qr`, `svd`, `lu`, `cholesky`, `symmetric_eigen`.
2. Current implementation strategy: iterate over `Axis(0)` and call per-matrix decomposition API.
3. Acceleration model today:
   - `Provider` may accelerate each per-matrix decomposition.
   - No dedicated batched decomposition GPU kernels today.

### Kernel-level batched APIs (dispatch-backed)

1. `matrix::batched_matmat*` and `matrix::batched_row_matvec*` are backend-dispatched.
2. GPU path currently supports these via `accelerator-wgpu` kernels with CPU fallback.

### Relevance map (locked)

1. **GPU-relevant batch paths (current):**
   - `matrix::batched_matmat*`
   - `matrix::batched_row_matvec*`
   - tensor N-D batched last-two matmul kernels (dispatch-backed)
2. **Provider-only batch paths (current):**
   - `batched::{qr, svd, lu, cholesky, symmetric_eigen}` (slice loop + per-slice decomposition call)
3. **Implication for V2:**
   - `V2-004` MAGMA work targets decomposition domains and will improve provider-backed batch APIs transitively (per-slice acceleration) before any dedicated batched decomposition GPU kernels are introduced.

## Runtime Size Policy (Current V2 Pass)

Policy is centralized in `accelerator::policy` and used by `GpuBackend` dispatch implementations.

1. Uses operation-specific thresholds (dense, sparse, vector, triangular, tensor).
2. Decides whether to attempt GPU or stay on CPU for small workloads.
3. Preserves existing explicit CPU fallback behavior for GPU-attempt failures.
4. Supports environment overrides for fast tuning during remote benchmarking.

### Remote release-profile baseline (RTX 4090, Vulkan)

1. Measured via `crates/nabled/tests/gpu_perf_probe.rs` on remote host.
2. Crossover for dense square `matmat` in release profile is approximately near `N ~= 900` (CPU generally faster below this range on this setup).
3. Policy defaults were updated conservatively to avoid over-attempting GPU on small/medium workloads; callers can override via `NABLED_GPU_MIN_*` env vars during tuning runs.

### Remote runbook

1. Provision/bootstrap host:
   - `scripts/gpu_v2_remote_prepare.sh <host>`
2. Run release-profile probe sweep:
   - `scripts/gpu_v2_remote_probe.sh <host>`
3. Run MAGMA correctness/perf verification bundle:
   - `scripts/gpu_v2_remote_magma_verify.sh <host>`
4. For headless Vulkan in containerized environments:
   - use EGL ICD (`libEGL_nvidia.so.0`) via `VK_ICD_FILENAMES` and set `XDG_RUNTIME_DIR`.

## V2-005 Result Snapshot (RTX 4090 Remote)

Artifacts:

1. `coverage/gpu-v2/magma/verification-4090.log`
2. `coverage/gpu-v2/magma/bench/openblas-system-summary-4090.json`
3. `coverage/gpu-v2/magma/bench/magma-system-summary-4090.json`

High-level outcomes:

1. MAGMA feature build/check/clippy and correctness suites pass on remote NVIDIA host.
2. Backend capability report generation under `magma-system` succeeds.
3. Provider benchmark comparison is captured for the same benchmark surface (`289` common entries).
4. The provider comparison shows mixed performance by domain/shape and confirms this should feed `K-005` outlier optimization, not a single global gate.

## Planned Checkpoints

1. `C1` Initial policy baseline check:
   - Validate no regressions in correctness.
   - Validate that small-shape calls avoid unnecessary GPU attempts.
2. `C2` Batched audit closure:
   - Record exact GPU-relevant batched paths and decomposition gaps.
3. `C3` MAGMA domain closure:
   - Wire targeted decomposition domains to MAGMA provider path and verify parity.

## Notes

1. This tracker is additive; v1 docs remain the historical baseline.
2. Once v2 work stabilizes, v1-only execution sections can be moved to an archive doc set.
