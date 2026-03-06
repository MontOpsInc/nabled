# GPU V2 Tracker

Last updated: 2026-03-06 (V2-009 MAGMA sparse/mixed-precision assessment complete)

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
| `V2-001` | Remote GPU verification environment | Completed | Vast 4090 runbook is scripted via canonical entrypoint (`scripts/gpu_remote.sh`) and validated via tmux-driven remote execution. |
| `V2-002` | Batched surface audit and GPU relevance map | Completed | Decomposition batch APIs are provider-accelerated per-slice loops; kernel-level batch APIs are backend-dispatched and GPU-capable (`batched_matmat`, `batched_row_matvec`). |
| `V2-003` | Runtime workload policy (small vs large) for GPU backend routing | Completed | Centralized `accelerator::policy` now gates all GPU backend-dispatched kernels with env-overridable thresholds tuned from remote 4090 release-profile measurements. |
| `V2-004` | MAGMA CUDA provider integration (scaffold + domain wiring) | Completed | MAGMA provider paths are wired for LU/Cholesky/QR/SVD/symmetric eigen, with provider-safe scalar bounds propagated across dependent domains. |
| `V2-005` | MAGMA verification matrix + benchmark parity report | Completed | Remote RTX 4090 run completed with correctness/capability artifacts and provider benchmark summaries captured locally. |
| `V2-006` | MAGMA provider breadth expansion (complex + additional decomposition domains) | Completed | Complex MAGMA provider paths now cover LU, Cholesky, QR, SVD, and non-symmetric eigen decomposition, with compile-time dispatch preference (`magma-system` > `lapack-provider` > internal) and explicit shape/error contracts. |
| `V2-007` | MAGMA-native batched decomposition kernels | Completed | `nabled-linalg::batched` now uses MAGMA-native batched LU/Cholesky/QR for real-valued provider paths under `magma-system`; SVD/symmetric-eigen remain per-slice provider loops due no equivalent MAGMA batched kernels for current API contracts. |
| `V2-008` | MAGMA sparse path assessment and integration plan | Completed | MAGMA sparse API fit was assessed on RTX 4090 host (`magmasparse_*` headers + exported symbols); integration plan is locked around an explicit `magma_sparse` FFI layer and opt-in sparse solve acceleration phases. |
| `V2-009` | MAGMA mixed-precision and iterative-refinement opportunities | Completed | MAGMA mixed-precision kernels were verified available (`magma_dsgesv_gpu`, `magma_dsgesv_iteref_gpu`, `magma_zcgesv_gpu`); integration plan is locked around opt-in mixed-precision solve APIs with explicit convergence/error contracts. |

## Batched Surface Snapshot (Current)

### Decomposition-level batched APIs (`nabled-linalg::batched`)

1. `qr`, `svd`, `lu`, `cholesky`, `symmetric_eigen`.
2. Current implementation strategy:
   - MAGMA-native batched kernels are used where available (`lu`, `cholesky`, `qr` under `magma-system`).
   - Remaining domains use per-slice provider loops (`svd`, `symmetric_eigen`).
3. Acceleration model:
   - `Provider` accelerates either batched decomposition kernels directly or each per-slice decomposition.

### Kernel-level batched APIs (dispatch-backed)

1. `matrix::batched_matmat*` and `matrix::batched_row_matvec*` are backend-dispatched.
2. GPU path currently supports these via `accelerator-wgpu` kernels with CPU fallback.

### Relevance map (locked)

1. **GPU-relevant batch paths (current):**
   - `matrix::batched_matmat*`
   - `matrix::batched_row_matvec*`
   - tensor N-D batched last-two matmul kernels (dispatch-backed)
2. **Provider-backed decomposition batch paths (current):**
   - Native MAGMA batched kernels: `batched::{lu, cholesky, qr}` (real-valued, `magma-system`)
   - Per-slice provider loops: `batched::{svd, symmetric_eigen}`
3. **Implication for V2:**
   - `V2-007` is complete for decomposition kernels currently available in MAGMA while preserving explicit contracts for non-batched domains.

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

1. Provision/bootstrap host + tmux session:
   - `scripts/gpu_remote.sh up <host>`
2. Run release-profile probe sweep:
   - `scripts/gpu_remote.sh one <host> gpu-probe`
3. Run MAGMA correctness/perf verification bundle:
   - `scripts/gpu_remote.sh one <host> magma-verify`
4. Run MAGMA capability scan (sparse + mixed precision symbols/headers):
   - `scripts/gpu_remote.sh one <host> magma-capability`
5. For headless Vulkan in containerized environments:
   - use EGL ICD (`libEGL_nvidia.so.0`) via `VK_ICD_FILENAMES` and set `XDG_RUNTIME_DIR`.
6. Pre-baked image optimization:
   - if host image has `/etc/nabled/nvidia-image`, `gpu_remote_prepare.sh` skips redundant apt/rust bootstrap.
7. Root/pre-baked execution reliability:
   - tmux job runners export a stable toolchain PATH including both `/home/agent/.cargo/bin` and `/root/.cargo/bin`.

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

## V2-006 Result Snapshot

1. Complex MAGMA provider kernels are implemented in `crates/nabled-linalg/src/provider/magma.rs` for:
   - LU solve/inverse/determinant
   - Cholesky decompose/solve/inverse
   - QR decomposition
   - SVD decomposition
   - Non-symmetric complex eigen decomposition
2. Domain dispatch is now normalized to prefer MAGMA when enabled:
   - `magma-system` -> MAGMA provider path
   - else `lapack-provider` -> LAPACK provider path
   - else -> internal implementation
3. Contract surface is explicit:
   - unsupported shape/provider constraints are returned as typed domain errors
   - no runtime provider selection; dispatch remains compile-time gated
4. Validation status:
   - `magma-system` strict clippy/check is green
   - repository quality gates (`just checks`) remain green after integration

## V2-008 Result Snapshot

Remote assessment outcomes (RTX 4090 host):

1. `magmasparse_*` headers and sparse symbols are present (`libmagma_sparse.so` + `magmasparse_{s,d,c,z}.h`).
2. Sparse entrypoints use MAGMA sparse domain types (`magma_[sdcz]_matrix`, `magma_[sdcz]_solver_par`, `magma_[sdcz]_preconditioner`) rather than raw CSR slices.
3. This is not a drop-in match for nabled sparse APIs (`CsrMatrix` / `CsrMatrixView`) and requires an explicit interop layer.

Integration plan (locked):

1. Add a dedicated `provider::magma_sparse` FFI boundary that owns MAGMA sparse struct lifecycle and queue management.
2. Phase 1 target: opt-in sparse matvec / sparse-dense matmat acceleration paths where contracts are clear and outputs remain deterministic.
3. Phase 2 target: iterative sparse solves/preconditioners (`cg/gmres/bicgstab`-family) with explicit tolerance/iteration contracts and typed convergence errors.
4. Keep provider/backend orthogonality: no hidden runtime provider switching and no hidden dense conversions.

## V2-009 Result Snapshot

Remote assessment outcomes (RTX 4090 host):

1. Mixed-precision dense solve kernels are available in MAGMA:
   - `magma_dsgesv_gpu`
   - `magma_dsgesv_iteref_gpu`
   - `magma_zcgesv_gpu`
2. Header-level contracts expose explicit refinement iteration/error outputs (`iter`, `info`) and dedicated mixed work buffers.

Integration plan (locked):

1. Introduce opt-in mixed-precision solve APIs first (do not silently replace existing `f64`/`Complex64` solve behavior).
2. Surface refinement metadata explicitly (`iterations`, `converged`, `fallback-used`) so callers can choose policy.
3. Map MAGMA `info`/iteration outcomes into typed domain errors; do not hide non-convergence.
4. Benchmark against current MAGMA double-precision solve baselines before enabling in default provider flows.

## MAGMA Expansion Scope (Post V2-005)

This is the explicit capture of MAGMA-oriented scope closed in v2:

1. Introduce MAGMA-native batched decomposition paths to reduce per-slice overhead in `nabled-linalg::batched`.
2. Assess MAGMA sparse APIs for fit with nabled sparse domain contracts (`CSR/CSC/COO`, preconditioners, solve reuse).
3. Evaluate mixed-precision + iterative-refinement paths as opt-in high-performance workflows.

All expansion items must preserve:

1. Provider/backend orthogonality.
2. Explicit API contracts (no hidden fallback surprises).
3. Existing quality gates and benchmark tracking discipline.

## Planned Checkpoints

1. `C1` Initial policy baseline check:
   - Validate no regressions in correctness.
   - Validate that small-shape calls avoid unnecessary GPU attempts.
2. `C2` Batched audit closure:
   - Record exact GPU-relevant batched paths and decomposition gaps.
3. `C3` MAGMA domain closure:
   - Completed: targeted decomposition domains are wired to MAGMA provider path and validated via strict `magma-system` clippy/check plus repository gates.

## Notes

1. This tracker is additive; v1 docs remain the historical baseline.
2. Once v2 work stabilizes, v1-only execution sections can be moved to an archive doc set.
