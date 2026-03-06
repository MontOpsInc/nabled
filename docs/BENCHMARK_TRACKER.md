# Benchmark Tracker

Last updated: 2026-03-06

## Purpose

This document is the execution tracker for benchmark-driven performance work.

It answers:

1. What we benchmark.
2. How we benchmark.
3. What we benchmark against.
4. Which scoped chunk is currently under optimization.

## Benchmark Contract

1. Benchmarks are run from `crates/nabled/benches/*_benchmarks.rs`.
2. Aggregation/reporting is produced by `crates/nabled/src/bin/benchmark_report.rs`.
3. Artifacts land in `coverage/benchmarks/`:
   - `summary.json`
   - `summary.csv`
   - `regressions.md`
   - baseline artifacts in `coverage/benchmarks/baseline/`
4. Use quick smoke runs for iteration; use full runs before final optimization signoff.
5. Track by chunk, not by ad hoc function.

## Chunk Model

A chunk is defined by:

1. Domain scope.
2. Execution config (`Provider`, `Backend`, dtype).
3. Benchmark suites and operations.
4. Competitor set.
5. Current measured status.

## Linalg Chunk Plan

| Chunk ID | Scope | Execution Config | Suites | Current Comparator Coverage | Status |
|---|---|---|---|---|---|
| `L-CPU-NATIVE-DECOMP` | Dense decompositions/solves | internal provider, CPU backend, `f64` (+ complex where exposed) | `svd`, `qr`, `lu`, `cholesky`, `eigen`, `schur`, `triangular` | broad (`faer_direct` for most + `schur` manual baseline) | In progress (measured) |
| `L-CPU-NATIVE-DENSE` | Dense matrix/vector kernels | internal provider, CPU backend, `f64` | `matrix`, `vector`, `orthogonalization`, `matrix_functions`, `polar` | mixed (`matrix`/`vector` have ndarray baselines; others mostly none) | In progress (measured) |
| `L-CPU-NATIVE-SPARSE` | Sparse kernels + sparse solves | internal provider, CPU backend, `f64` | `sparse` | partial (ndarray dense baseline) | In progress (measured) |
| `L-CPU-NATIVE-TENSOR` | Tensor/cube kernels | internal provider, CPU backend, `f64` (+ complex where exposed) | `tensor` | partial (manual baseline subset) | Planned |
| `L-CPU-NATIVE-ACCEL` | Accelerator CPU paths | internal provider, CPU backend | `accelerator` | partial (manual baseline) | Planned |
| `L-CPU-PROVIDER-DECOMP` | Provider-backed decompositions | `openblas-system` provider, CPU backend | same as `L-CPU-NATIVE-DECOMP` | same comparator gaps as native unless expanded | Planned |
| `L-CPU-PROVIDER-MAGMA` | Provider-backed decompositions (NVIDIA MAGMA) | `magma-system` provider, CPU backend | same as `L-CPU-NATIVE-DECOMP` | paired against `openblas-system` provider summaries | Measured |
| `L-GPU-WGPU-F32` | GPU kernel paths | internal/provider orthogonal, `accelerator-wgpu`, `f32` scope | `accelerator` (`accelerator_nabled_gpu_cpu_f32`, `accelerator_nabled_gpu_wgpu_f32`) + tensor GPU-covered paths | expanded (`f32` CPU comparator + GPU groups) | In progress (measured + optimization active) |

## Active Round Scope

1. Active optimization/expansion round is GPU-only.
2. Metal-specific backend work is explicitly deferred.
3. SIMD-focused CPU optimization is explicitly deferred.
4. V2 execution sequencing for GPU/provider expansion is tracked in `docs/GPU_V2_TRACKER.md`.

## Current Harness Audit (Linalg)

| Suite | Group(s) | Operations | Comparator in Active Suite |
|---|---|---|---|
| `svd_benchmarks` | `svd_nabled_ndarray`, `svd_competitor_faer_direct` | `full_svd`, `truncated_svd`, `full_svd_complex` | `faer_direct` (real paths) |
| `qr_benchmarks` | `qr_nabled_ndarray`, `qr_competitor_faer_direct` | `qr`, `least_squares` | `faer_direct` |
| `lu_benchmarks` | `lu_nabled_ndarray`, `lu_competitor_faer_direct` | `decompose`, `solve`, `determinant` | `faer_direct` |
| `cholesky_benchmarks` | `cholesky_nabled_ndarray`, `cholesky_competitor_faer_direct` | `decompose`, `solve`, `inverse` | `faer_direct` |
| `eigen_benchmarks` | `eigen_nabled_ndarray`, `eigen_competitor_faer_direct` | `symmetric`, `generalized` | `faer_direct` |
| `schur_benchmarks` | `schur_nabled_ndarray`, `schur_competitor_manual` | `compute_schur`, `manual_qr_iteration` | manual baseline |
| `triangular_benchmarks` | `triangular_nabled_ndarray`, `triangular_competitor_faer_direct` | `solve_lower`, `solve_upper` | `faer_direct` |
| `matrix_benchmarks` | `matrix_nabled_ndarray`, `matrix_competitor_ndarray` | `matvec`, `matmat`, batched variants | ndarray baseline |
| `vector_benchmarks` | `vector_nabled_ndarray`, `vector_competitor_ndarray` | `dot`, `cosine_similarity`, pairwise variants | ndarray baseline |
| `sparse_benchmarks` | `sparse_nabled_ndarray`, `sparse_competitor_ndarray` | sparse core + solve/reuse paths | ndarray dense baseline |
| `tensor_benchmarks` | `tensor_nabled_ndarray`, `tensor_competitor_manual` | cube + `ArrayD` ops | manual baseline |
| `accelerator_benchmarks` | `accelerator_nabled_ndarray`, `accelerator_competitor_manual` | CPU accelerator kernel variants | manual baseline |
| `matrix_functions_benchmarks` | `matrix_functions_nabled_ndarray` | exp/log/power paths | none |
| `orthogonalization_benchmarks` | `orthogonalization_nabled_ndarray` | GS variants | none |
| `polar_benchmarks` | `polar_nabled_ndarray` | polar (real/complex) | none |

Notes:

1. `benchmark_report` now classifies active decomposition `faer_direct` competitor groups (`svd`, `qr`, `lu`, `cholesky`, `eigen`, `triangular`).
2. Complex decomposition comparator coverage is still partial.

## Chunk Run: `L-CPU-NATIVE-DECOMP`

Execution config:

1. Provider: internal/default (no `openblas-system`).
2. Backend: CPU.
3. Dtype: `f64` plus complex cases where the suite currently includes them.

Commands used:

```bash
cargo bench -p nabled --bench svd_benchmarks -- --quick
cargo bench -p nabled --bench qr_benchmarks -- --quick
cargo bench -p nabled --bench lu_benchmarks -- --quick
cargo bench -p nabled --bench cholesky_benchmarks -- --quick
cargo bench -p nabled --bench eigen_benchmarks -- --quick
cargo bench -p nabled --bench schur_benchmarks -- --quick
cargo bench -p nabled --bench triangular_benchmarks -- --quick
cargo run -p nabled --bin benchmark_report
```

Scoped extraction for this chunk uses benchmark IDs:

1. `svd_nabled_ndarray/(full_svd|truncated_svd|full_svd_complex)/<size>`
2. `qr_nabled_ndarray/(qr|least_squares)/<size>`
3. `lu_nabled_ndarray/(decompose|solve|determinant)/<size>`
4. `cholesky_nabled_ndarray/(decompose|solve|inverse)/<size>`
5. `eigen_nabled_ndarray/(symmetric|generalized)/<size>`
6. `schur_nabled_ndarray/compute_schur/square-<n>x<n>`
7. `schur_competitor_manual/manual_qr_iteration/square-<n>x<n>`
8. `triangular_nabled_ndarray/(solve_lower|solve_upper)/square-<n>x<n>`

### Results Snapshot (Geometric Mean of Median ns by Domain)

| Domain | Competitor | Cases | Geomean Median ns |
|---|---:|---:|---:|
| `cholesky` | `none` | 9 | 16,471.621 |
| `eigen` | `none` | 6 | 220,580.955 |
| `lu` | `none` | 9 | 38,557.027 |
| `qr` | `none` | 6 | 134,688.549 |
| `schur` | `none` | 3 | 2,713,887.815 |
| `schur` | `manual_baseline` | 2 | 2,016,187.616 |
| `svd` | `none` | 8 | 872,576.363 |
| `triangular` | `none` | 6 | 1,210.066 |

### Comparator Ratio Snapshot (`schur`)

| Shape | nabled median ns | manual baseline median ns | nabled/manual |
|---|---:|---:|---:|
| `square-16x16` | 539,522.627 | 857,301.430 | 0.629 |
| `square-32x32` | 3,547,721.375 | 4,741,637.375 | 0.748 |
| `square-48x48` | 10,442,789.062 | n/a | n/a |

Interpretation:

1. `schur` is faster than the current manual baseline on measured overlapping shapes.
2. Additional decomposition comparator coverage was added in a later iteration (below) and should be used for parity claims.

## Chunk Run: `L-CPU-NATIVE-DENSE`

Execution config:

1. Provider: internal/default (no `openblas-system`).
2. Backend: CPU.
3. Dtype: `f64` (+ complex where suite currently includes it).

Commands used:

```bash
cargo bench -p nabled --bench matrix_benchmarks -- --quick
cargo bench -p nabled --bench vector_benchmarks -- --quick
cargo bench -p nabled --bench orthogonalization_benchmarks -- --quick
cargo bench -p nabled --bench matrix_functions_benchmarks -- --quick
cargo bench -p nabled --bench polar_benchmarks -- --quick
cargo run -p nabled --bin benchmark_report
```

Data hygiene for extraction:

```bash
awk -F, 'NR==1{print;next}{rows[$15]=$0;ord[++n]=$15} END{for(i=1;i<=n;i++){id=ord[i]; if(!seen[id]++) print rows[id]}}' \
  coverage/benchmarks/summary.csv > /tmp/summary_dedup.csv
```

### Results Snapshot (Geometric Mean of Median ns by Domain)

| Domain | Competitor | Cases | Geomean Median ns |
|---|---:|---:|---:|
| `matrix` | `none` | 8 | 141,231.971 |
| `matrix` | `ndarray_baseline` | 8 | 141,632.327 |
| `matrix_functions` | `none` | 25 | 48,991.031 |
| `orthogonalization` | `none` | 6 | 290,512.070 |
| `polar` | `none` | 5 | 93,229.244 |
| `vector` | `none` | 10 | 2,132.903 |
| `vector` | `ndarray_baseline` | 8 | 383.721 |

### Comparator Ratio Snapshot

Matrix (`nabled / ndarray_baseline`, matched op+size):

1. Geomean ratio: `0.997` (near parity).
2. `matvec`: `0.967` (128), `0.951` (256), `1.004` (512).
3. `matmat`: `0.974` (128), `1.057` (256), `0.998` (512).

Vector (`nabled / ndarray_baseline`, matched op+size):

1. `cosine_similarity`: `1.083` (128), `1.028` (256), `1.052` (512) -> near parity.
2. `dot`: `4.692` (128), `4.922` (256), `6.422` (512) -> clear regression hotspot.
3. `pairwise_l2`: `1.040` (32x32), `1.009` (64x64) -> near parity.

Interpretation:

1. Dense matrix kernels are effectively at parity with ndarray baseline in current smoke sizes.
2. Vector `dot` path is the dominant dense-kernel regression and should be first optimization target.
3. `orthogonalization`, `matrix_functions`, and `polar` still need comparator expansion for ecosystem parity claims.

### Iteration: `vector::dot` optimization pass

Change:

1. `dot_serial` now delegates to ndarray `dot` instead of manual scalar loop.

Result (`vector` nabled vs ndarray baseline, matched op+size):

1. Before geomean ratio: `1.917` (with `dot` at `~4.7x` to `~6.4x` slower).
2. After geomean ratio: `1.051`.
3. `dot` now: `1.049` (128), `1.038` (256), `1.109` (512).

Interpretation:

1. Major dense hotspot closed to near parity.
2. Remaining vector gap is now minor and no longer the top benchmark risk.

## Decomposition Comparator Expansion Iteration

Execution config:

1. Provider: internal/default.
2. Backend: CPU.
3. Dtype: `f64` for `faer_direct` comparator paths.

Commands used:

```bash
cargo bench -p nabled --bench svd_benchmarks -- --quick
cargo bench -p nabled --bench qr_benchmarks -- --quick
cargo bench -p nabled --bench lu_benchmarks -- --quick
cargo bench -p nabled --bench cholesky_benchmarks -- --quick
cargo bench -p nabled --bench eigen_benchmarks -- --quick
cargo bench -p nabled --bench triangular_benchmarks -- --quick
cargo run -p nabled --bin benchmark_report
```

Ratio snapshot (`nabled / faer_direct`, matched decomposition IDs):

1. `svd` domain geomean: `1.450`.
2. `qr` domain geomean: `3.412`.
3. `lu` domain geomean: `3.019`.
4. `cholesky` domain geomean: `4.012` (before targeted inverse optimization pass).
5. `eigen` domain geomean: `2.567`.
6. `triangular` domain geomean: `1.490`.

Top measured outliers:

1. `cholesky::inverse`: `~9.9x` (16), `~21.0x` (32), `~54.3x` (64).
2. `qr` (96): `~5.8x` (`qr`), `~5.4x` (`least_squares`).
3. `lu` (96): `~4.0x` (`decompose`), `~3.7x` (`solve`), `~4.2x` (`determinant`).
4. `eigen::symmetric`: `~3.4x` to `~4.0x`.

### Iteration: `cholesky::inverse` optimization pass

Change:

1. Internal Cholesky inverse path now reuses a single factorization and solves from that factor, instead of repeatedly re-entering solve/decompose pathways per inverse column.

Result (`nabled / faer_direct`, same decomposition chunk):

1. `cholesky` domain geomean improved from `4.012` -> `2.409`.
2. `cholesky::inverse` ratios improved:
   - size 16: `9.906` -> `3.400`
   - size 32: `20.961` -> `5.075`
   - size 64: `54.271` -> `7.789`

Interpretation:

1. This closes a major algorithmic inefficiency and materially reduces one of the largest outliers.
2. Remaining decomposition hotspots are now led by `qr`, `lu`, and `eigen::symmetric`.

## Chunk Run: `L-CPU-NATIVE-SPARSE`

Execution config:

1. Provider: internal/default (no `openblas-system`).
2. Backend: CPU.
3. Dtype: `f64`.

Commands used:

```bash
cargo bench -p nabled --bench sparse_benchmarks -- --quick
cargo run -p nabled --bin benchmark_report
```

### Results Snapshot (Geometric Mean of Median ns by Domain)

| Domain | Competitor | Cases | Geomean Median ns |
|---|---:|---:|---:|
| `sparse` | `none` | 96 | 21,534.342 |
| `sparse` | `ndarray_baseline` | 3 | 10,209.395 |

### Comparator Ratio Snapshot

Sparse matvec vs dense ndarray baseline (`nabled / dense_baseline`, matched size):

1. Geomean ratio: `0.080`.
2. `csr_matvec`: `0.185` (128), `0.083` (256), `0.038` (512).
3. `csr_matvec_into`: `0.175` (128), `0.074` (256), `0.036` (512).

Interpretation:

1. Current sparse baseline is intentionally coarse (dense matvec), so this ratio is directional only.
2. Sparse domain has broad internal coverage but lacks high-fidelity external sparse-library baselines for many operations.
3. Next sparse benchmarking work should prioritize comparator quality over additional case count.

## Audit Findings

1. Benchmark chunking by execution axis is the right approach (`Provider` and `Backend` materially change interpretation).
2. Comparator coverage is uneven:
   - strong for `matrix`, `vector`, some `sparse`, some `accelerator`.
   - weak for decomposition-heavy linalg where ecosystem parity matters most.
3. Existing benchmark artifacts can include historical entries from old run directories; chunk extraction must be strict by benchmark ID patterns.

## Next Benchmarking Actions

1. Drive `L-GPU-WGPU-F32` as the primary optimization chunk:
   - keep CPU comparator (`accelerator_nabled_gpu_cpu_f32`) and GPU group (`accelerator_nabled_gpu_wgpu_f32`) locked for regression tracking.
2. Continue GPU result split by operation family:
   - `dense` (`matmat`, `matvec`, batched variants),
   - `vector` (`dot`, pairwise kernels),
   - `tensor` (contract/reduction/batched matmul),
   - `sparse` (when phase-1 sparse GPU kernels land).
3. For each GPU-native op, track:
   - correctness parity vs CPU baseline,
   - median/percentile timing ratio,
   - fallback incidence (should trend to zero for in-scope GPU-native ops).
4. Keep CPU decomposition hotspot optimization deferred for this round.
5. Keep SIMD-driven optimization deferred for this round.

### Remote V2 Baseline (RTX 4090, release profile)

1. Remote sweep run via `scripts/gpu_remote.sh one <host> gpu-probe` with `accelerator-wgpu` and Vulkan.
2. Dense square `matmat` crossover (CPU vs direct GPU path) on this setup is approximately near `N ~= 900`.
3. This baseline informed `accelerator::policy` default thresholds so GPU dispatch avoids small/medium workload regressions by default.

## Chunk Run: `L-CPU-PROVIDER-MAGMA` (Remote RTX 4090)

Execution config:

1. Providers compared: `openblas-system` vs `magma-system`.
2. Backend: CPU backend orchestration (provider-sensitive decomposition internals).
3. Feature path: `--no-default-features --features <provider>`.
4. Host: Vast.ai RTX 4090 environment.

Commands used:

```bash
scripts/gpu_remote.sh up <host>
scripts/gpu_remote.sh one <host> magma-verify
# provider comparison pass:
NABLED_PROVIDER_BENCH_FEATURES=openblas-system just -f .justfile bench-smoke-report-provider
NABLED_PROVIDER_BENCH_FEATURES=magma-system    just -f .justfile bench-smoke-report-provider
```

Artifacts:

1. `coverage/gpu-v2/magma/verification-4090.log`
2. `coverage/gpu-v2/magma/bench/openblas-system-summary-4090.json`
3. `coverage/gpu-v2/magma/bench/magma-system-summary-4090.json`

Summary:

1. Both provider runs produced complete benchmark summaries (`289` common benchmark entries).
2. Median ratio (`magma/openblas`) across all entries is near parity (`~0.997`), with domain-specific outliers concentrated in tiny-shape decomposition cases where fixed provider overhead dominates.
3. For decomposition-heavy domains, MAGMA is mixed by operation/size and should be tuned via outlier-ranked follow-up (`K-005`) rather than treated as globally faster/slower.

## Optimization Handoff Notes (Cholesky, 2026-03-03)

Use these notes when resuming point-by-point parity work.

### What was learned

1. `release-lto` profile materially changes ranking and should be treated as canonical for parity work.
2. `coverage/benchmarks/summary.csv` can contain duplicate `full_id` rows across repeated runs; parity extraction should dedup by `full_id` before ratio calculations.
3. Criterion `new/estimates.json` under `target/criterion/*` is the cleanest source for a single-run comparison.
4. The largest gains in Cholesky came from:
   - contiguous-slice kernels (`as_slice_memory_order[_mut]`),
   - avoiding full input copy where not needed,
   - reducing bounds checks in O(n^3) loops via documented `unsafe`,
   - loop unrolling in inner dot-product-like regions.
5. Remaining Cholesky gaps are concentrated at larger sizes (`32`, `64`) and especially `inverse/64`.

### Practical extraction patterns

Dedup summary CSV by `full_id`:

```bash
awk -F, 'NR==1{print;next}{rows[$15]=$0;ord[++n]=$15} END{for(i=1;i<=n;i++){id=ord[i]; if(!seen[id]++) print rows[id]}}' \
  coverage/benchmarks/summary.csv > /tmp/summary_dedup.csv
```

Single-run ratio extraction (criterion):

```bash
python3 - <<'PY'
import json, math
from pathlib import Path
root=Path('target/criterion')
ops=['decompose','solve','inverse']
sizes=[16,32,64]
rat=[]
for op in ops:
    for size in sizes:
        n=root/f'cholesky_nabled_ndarray/{op}/{size}/new/estimates.json'
        f=root/f'cholesky_competitor_faer_direct/{op}/{size}/new/estimates.json'
        nm=json.loads(n.read_text())['median']['point_estimate']
        fm=json.loads(f.read_text())['median']['point_estimate']
        r=nm/fm
        rat.append(r)
        print(f'{op:10s} {size:>3}: ratio={r:6.3f}')
print(f'geomean ratio={math.exp(sum(math.log(x) for x in rat)/len(rat)):.3f}')
PY
```

### Resume target

1. Keep pushing Cholesky until geomean `nabled/faer_direct` is near `1.0`.
2. After Cholesky parity, continue in planned order: `qr` -> `lu` -> `eigen::symmetric`.
