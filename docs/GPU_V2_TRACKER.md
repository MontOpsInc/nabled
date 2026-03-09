# GPU V2 Tracker

Last updated: 2026-03-09 (K-005 LTO proof-pack overlay rerun on RTX 4090)

## Purpose

Track the v2 direction for GPU/provider expansion without losing v1 stabilization history.

This tracker is focused on:

1. Batch-oriented GPU workload coverage.
2. Runtime GPU routing policy for small vs large workloads.
3. MAGMA provider integration (CUDA first).
4. Verification harness for rapid remote iteration.
5. MAGMA signoff evidence at per-API granularity (`docs/MAGMA_SIGNOFF.md`).

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
| `V2-007` | MAGMA-native batched decomposition kernels | Completed | `nabled-linalg::batched` uses MAGMA-native batched LU/Cholesky/QR; `batched::svd*` and `batched::symmetric_eigen*` now attempt MAGMA per-slice routes in `M*` builds with policy/strict gating, while native batched SVD/eigen symbols are absent in current MAGMA runtime (captured in `coverage/gpu-v2/magma/capability-batched-symbols-20260307.log`). |
| `V2-008` | MAGMA sparse path assessment and integration plan | Completed | Assessment is complete and both sparse phases are implemented: dedicated `provider::magma_sparse` FFI lifecycle, phase-1 sparse kernels (`matvec`, sparse-dense `matmat`), and phase-2 iterative/preconditioned solves (`CG`, `PCG-Jacobi`, `GMRES`, `BiCGSTAB`, plus `ILU0`-preconditioned `GMRES`/`BiCGSTAB`) for `f32`/`f64` `i32` CSR views. |
| `V2-009` | MAGMA mixed-precision and iterative-refinement opportunities | Completed | Availability was verified and phase-1/phase-2 implementations are landed: opt-in mixed-precision LU solve APIs plus Sylvester/Lyapunov mixed APIs (`solve_sylvester_mixed_*`, `solve_lyapunov_mixed_*`) now delegate to MAGMA refinement paths with explicit refinement/error contracts. |

## V2 Completion Contract (Done Definition)

V2 is considered complete only when all items below are true:

1. Provider precedence is normalized across decomposition domains: `magma-system` > `lapack-provider` > internal (including real and complex QR paths).
2. Batched decomposition routing is dynamic for MAGMA-supported domains (`lu`, `cholesky`, `qr`) and uses both batch cardinality and matrix-shape/work signals, with explicit env overrides for tuning.
3. Routing-policy overhead is bounded: policy/env lookups are cached and do not repeatedly parse environment variables on hot paths.
4. Feature-matrix behavior is validated for relevant combinations (`internal`, `lapack-provider`, `magma-system`, `lapack-provider + magma-system`, plus accelerator permutations where applicable).
5. Post-routing provider benchmark comparison is rerun on remote NVIDIA host using canonical route-quality overlay comparison (`openblas-system` baseline vs `openblas-system+magma-system` overlay), and outlier tables are refreshed.
6. Docs and trackers remain synchronized (`GPU_V2_TRACKER`, `EXECUTION_TRACKER`, `STATUS`, and benchmark notes) with no ambiguity in current routing behavior.
7. Strict MAGMA verification exists and is reproducible (`scripts/gpu_remote.sh one <host> magma-strict-verify`), forcing decomposition thresholds low and failing fast on provider runtime errors (`NABLED_MAGMA_STRICT=1`).

## K-005 Progress Snapshot

1. Phase-1 complete: centralized small-shape decomposition MAGMA cutoff (`MagmaProviderPolicy::prefer_decomposition`) with env override.
2. Phase-2 complete: `lapack-provider + magma-system` combined feature matrix is compile/clippy clean for `nabled-linalg`, with cfg precedence conflicts resolved across decomposition-adjacent domains.
3. Phase-3 complete: dynamic batched decomposition routing is active for MAGMA-supported domains (`lu`, `cholesky`, `qr`) via `MagmaProviderPolicy::prefer_batched_decomposition`.
4. Phase-4 complete: policy/env resolution is cached (`OnceLock`), remote strict/normal MAGMA verification both pass on RTX 4090, and provider benchmark rerun/outlier refresh is completed on the current routing snapshot.
5. Phase-5 complete: decomposition monitor/proof-pack harness now enforces persistent slowdown by both ratio and effect size (`ratio > 1.03` and `delta_ns > 5000`), alternates run order per repeat, and the latest LTO overlay rerun reports `persistent_regression_count = 0` (`stability-20260309T130811Z.json`).

## Batched Surface Snapshot (Current)

### Decomposition-level batched APIs (`nabled-linalg::batched`)

1. `qr`, `svd`, `lu`, `cholesky`, `symmetric_eigen`.
2. Current implementation strategy:
   - MAGMA-native batched kernels are used where available (`lu`, `cholesky`, `qr` under `magma-system`).
   - `svd` and `symmetric_eigen` attempt MAGMA via per-slice provider routes (policy-gated) because native batched SVD/eigen symbols are absent in current MAGMA runtime.
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
   - Policy-gated per-slice MAGMA routes: `batched::{svd, symmetric_eigen}` (`M*`, strict-fail aware)
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
   - `scripts/gpu_remote.sh one <host> magma-k005-monitor`
   - `scripts/gpu_remote.sh one <host> magma-proof-pack`
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

### V2-008 Follow-on Implementation (Phase 1)

1. Added dedicated sparse provider FFI boundary:
   - `crates/nabled-linalg/src/provider/magma_sparse.rs`
2. Added opt-in sparse MAGMA APIs (i32-indexed CSR view contract):
   - `sparse::matvec_magma_{f32,f64}_view`
   - `sparse::matvec_magma_{f32,f64}_view_into`
   - `sparse::matmat_dense_magma_{f32,f64}_view`
   - `sparse::matmat_dense_magma_{f32,f64}_view_into`
3. Added parity tests against internal sparse paths for all four operation/dtype combinations.
4. Scope was intentionally phase-1 at this checkpoint; phase-2 is now implemented in follow-on APIs.

### V2-008 Follow-on Implementation (Phase 2)

1. Added MAGMA-backed iterative sparse solve APIs for `i32`-indexed CSR views:
   - `conjugate_gradient_magma_f64_view`, `conjugate_gradient_magma_f32_view`
   - `pcg_jacobi_magma_f64_view`, `pcg_jacobi_magma_f32_view`
   - `gmres_magma_f64_view`, `gmres_magma_f32_view`
   - `bicgstab_magma_f64_view`, `bicgstab_magma_f32_view`
2. Added MAGMA-backed preconditioned sparse solve APIs:
   - `gmres_ilu0_magma_f64_view`, `gmres_ilu0_magma_f32_view`
   - `bicgstab_ilu0_magma_f64_view`, `bicgstab_ilu0_magma_f32_view`
3. Added parity coverage tests versus internal solver paths for `f64` and `f32` across SPD and non-symmetric systems.

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

### V2-009 Follow-on Implementation (Phase 1)

1. Landed in `nabled-linalg::lu`:
   - `solve_mixed_f64`, `solve_mixed_f64_view`
   - `solve_mixed_complex`, `solve_mixed_complex_view`
2. Return contract is explicit via `MixedSolveResult<T>`:
   - `solution`
   - `refinement_iterations`
3. Error mapping is explicit:
   - `convergence_failed` maps to `LUError::ConvergenceFailed`
   - missing MAGMA feature maps to explicit `InvalidInput` message
4. Phase-2 expansion now extends mixed-precision APIs into Sylvester/Lyapunov solvers.

### V2-009 Follow-on Implementation (Phase 2)

1. Landed in `nabled-linalg::sylvester`:
   - `solve_sylvester_mixed_f64`, `solve_sylvester_mixed_f64_view`
   - `solve_sylvester_mixed_complex`, `solve_sylvester_mixed_complex_view`
   - `solve_lyapunov_mixed_f64`, `solve_lyapunov_mixed_f64_view`
   - `solve_lyapunov_mixed_complex`, `solve_lyapunov_mixed_complex_view`
2. Return contract is explicit via `MixedSylvesterResult<T>`:
   - `solution`
   - `refinement_iterations`
3. Error mapping is explicit:
   - LU mixed convergence/singularity outcomes map to `SylvesterError::SingularSystem`
   - missing MAGMA feature maps to explicit `SylvesterError::InvalidInput` message

## K-005 Result Snapshot

1. Root-cause class for dominant MAGMA provider outliers was confirmed as tiny-shape fixed overhead.
2. A centralized MAGMA size policy gate is now applied before selecting MAGMA decomposition kernels:
   - `MagmaProviderPolicy::prefer_decomposition(rows, cols)`
   - default threshold: `min(rows, cols) >= 128`
   - runtime override: `NABLED_MAGMA_MIN_DECOMPOSITION_DIM=<usize>`
3. Phase-1 routing is wired in:
   - `lu` solve/inverse/determinant (real + complex),
   - `qr` decomposition provider paths (real + complex),
   - `svd` decomposition provider paths (real + complex).
4. Strict/normal validation is now complete on remote RTX 4090:
   - `scripts/gpu_remote.sh one <host> magma-verify` -> `rc=0`
   - `scripts/gpu_remote.sh one <host> magma-strict-verify` -> `rc=0`
   - strict log is clean for prior sparse CUDA context-noise signatures (`provider_alloc_failed`, memory-free error lines),
     and forced execution-matrix stderr hygiene (`cusparseCreate`, `cusparseSetStream`) is now closed in `docs/MAGMA_SIGNOFF.md`.
   - strict workflow now runs serialized tests (`RUST_TEST_THREADS=1`) and splits baseline correctness
     from forced strict execution-matrix checks to keep MAGMA signoff signal deterministic.
5. Post-routing provider benchmark rerun is complete (`openblas-system` vs `magma-system`) with refreshed outlier ranking artifacts:
   - `coverage/gpu-v2/magma/bench/openblas-system-summary-20260308T154226Z.json`
   - `coverage/gpu-v2/magma/bench/magma-system-summary-20260308T154226Z.json`
   - `coverage/gpu-v2/magma/bench/comparison-20260308T154226Z.md`
6. Refreshed decomposition-focused ratio snapshot (`magma/openblas`, nabled decomposition domains only):
   - common entries: `62`
   - median ratio: `~1.006`
   - p90 ratio: `~1.056`
   - `eigen::generalized(32)` is now parity-class (`~0.983x`) in this rerun,
   - largest remaining regressions in this run are concentrated in `cholesky::{inverse(32|64), solve(64)}` and `eigen::generalized(48)`.
   - run-to-run host variance remains visible, so K-005 outlier decisions should be based on repeated reruns.
7. MAGMA signoff expansion is complete at function granularity:
   - canonical route ledger: `docs/MAGMA_SIGNOFF.md`
   - one-row-per-public-function matrix: `docs/MAGMA_PUBLIC_API_MATRIX.md`
8. Composed-domain MAGMA signoff is now closed:
   - routed rows `MAG-D-030..MAG-D-043` (`schur`, `polar`, matrix-functions) are verified,
   - strict remote evidence: `job-20260307T203307Z.log` (`rc=0`).
9. K-005 follow-on composition pass is complete:
   - `magma-system` now composes with the lapack provider stack (`openblas-system`) for fallback quality,
   - `qr::solve_least_squares` and complex QR/SVD provider routes are MAGMA-first with lapack fallback in `magma+lapack` builds,
   - strict verification remains green after composition (`job-20260307T221821Z.log`, `rc=0`).
10. K-005 follow-on small/medium routing pass is implemented and remotely validated:
   - `MagmaProviderPolicy::prefer_decomposition` now requires both minimum dimension and minimum work (`rows*cols`) before routing to MAGMA,
   - new env override is available: `NABLED_MAGMA_MIN_DECOMPOSITION_WORK=<usize>`,
   - `lu` and `cholesky` provider routes now use MAGMA-first with lapack fallback in `magma+lapack` builds for non-eligible and runtime-fallback paths,
   - strict verification (`job-20260308T155035Z.log`, `rc=0`) and remote provider benchmark refresh (`comparison-20260308T154226Z.md`) are complete.
11. Latest pass fixed `magma-system` compile-matrix parity for eigen callers:
   - `matrix_functions` scalar bounds now match `lapack-provider + magma-system` expectations,
   - local clippy is green for `--no-default-features --features magma-system`,
   - remote strict verification remains green on the patched tree.
12. Decomposition-only repeated stability sweep is complete on the same host (`REPEATS=5`):
   - artifacts: `stability-20260308T162602Z.{json,md}` plus per-run comparison reports,
   - run medians remained near parity (`~0.997` to `~1.005`) with p90 in `~1.024` to `~1.038`,
   - no decomposition regression remains persistently >`1.03x` across recent repeated runs,
   - `K-005` is now in monitor mode; reopen targeted optimization only for regressions that persist across repeated batches.
13. Persistent-regression monitor gate is now scripted:
   - job: `scripts/remote_jobs/magma_k005_monitor_job.sh`
   - defaults: `REPEATS=5`, `PERSISTENT_RATIO_THRESHOLD=1.03`, `PERSISTENT_MIN_RUNS=4`
   - wrapper: `scripts/gpu_remote.sh one <host> magma-k005-monitor`

## MAGMA Expansion Scope (Post V2-005)

This is the explicit capture of MAGMA-oriented scope closed in v2:

1. Introduce MAGMA-native batched decomposition paths where available in `nabled-linalg::batched`; for unavailable kernels (`svd`/`symmetric_eigen`), lock explicit per-slice MAGMA routing with symbol-scan evidence.
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
