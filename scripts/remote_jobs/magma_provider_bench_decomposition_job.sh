#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

# Provider labels are used in artifact file names.
BASELINE_PROVIDER="${BASELINE_PROVIDER:-openblas-system}"
MAGMA_PROVIDER="${MAGMA_PROVIDER:-openblas-system+magma-system}"

# Provider feature sets drive `cargo bench --features ...`.
# Defaults compare baseline OpenBLAS vs OpenBLAS+MAGMA overlay so
# persistent slowdowns represent MAGMA routing quality, not missing
# provider coverage in a magma-only build.
BASELINE_PROVIDER_FEATURES="${BASELINE_PROVIDER_FEATURES:-openblas-system}"
MAGMA_PROVIDER_FEATURES="${MAGMA_PROVIDER_FEATURES:-openblas-system,magma-system}"
REPEATS="${REPEATS:-3}"
ARTIFACT_DIR="${ARTIFACT_DIR:-coverage/gpu-v2/magma/bench/decomposition}"
BATCH_TS="$(date -u +%Y%m%dT%H%M%SZ)"
BENCH_COMMAND="${BENCH_COMMAND:-bench-smoke-report-provider-decomposition}"
PERSISTENT_RATIO_THRESHOLD="${PERSISTENT_RATIO_THRESHOLD:-1.03}"
PERSISTENT_MIN_DELTA_NS="${PERSISTENT_MIN_DELTA_NS:-5000}"
PERSISTENT_MIN_RUNS="${PERSISTENT_MIN_RUNS:-${REPEATS}}"
ENFORCE_PERSISTENT_REGRESSIONS="${ENFORCE_PERSISTENT_REGRESSIONS:-0}"

# Keep decomposition stability runs deterministic: clear any ad hoc
# threshold/strict overrides that may have been used by previous jobs.
unset NABLED_MAGMA_MIN_DECOMPOSITION_DIM || true
unset NABLED_MAGMA_MIN_DECOMPOSITION_WORK || true
unset NABLED_MAGMA_VERIFY_FORCE || true
unset NABLED_MAGMA_STRICT || true

# Keep provider comparisons on consistent BLAS/OpenMP threading.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${ARTIFACT_DIR}"

run_provider_pass() {
  local provider_features="$1"
  local out_json="$2"

  rm -rf target/criterion crates/nabled/target/criterion
  rm -f coverage/benchmarks/summary.json
  NABLED_PROVIDER_BENCH_FEATURES="${provider_features}" just -f .justfile "${BENCH_COMMAND}"
  if [[ ! -s coverage/benchmarks/summary.json ]]; then
    echo "missing benchmark summary after provider pass: ${provider_features}" >&2
    exit 1
  fi
  cp coverage/benchmarks/summary.json "${out_json}"
}

compare_pass() {
  local baseline_json="$1"
  local magma_json="$2"
  local report_md="$3"
  python3 - <<'PY' "${baseline_json}" "${magma_json}" "${report_md}"
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
magma_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])

baseline = json.loads(baseline_path.read_text())["entries"]
magma = json.loads(magma_path.read_text())["entries"]

baseline_map = {
    entry["full_id"]: entry
    for entry in baseline
    if entry.get("competitor") == "none" and "_nabled_" in entry.get("group_id", "")
}
magma_map = {
    entry["full_id"]: entry
    for entry in magma
    if entry.get("competitor") == "none" and "_nabled_" in entry.get("group_id", "")
}

rows: list[dict[str, object]] = []
for full_id, magma_entry in magma_map.items():
    baseline_entry = baseline_map.get(full_id)
    if baseline_entry is None:
        continue
    baseline_ns = float(baseline_entry["mean_ns"])
    magma_ns = float(magma_entry["mean_ns"])
    if baseline_ns <= 0.0:
        continue
    rows.append(
        {
            "full_id": full_id,
            "domain": magma_entry.get("domain", ""),
            "operation": magma_entry.get("operation", ""),
            "dtype": magma_entry.get("dtype", ""),
            "size": magma_entry.get("size", ""),
            "ratio": magma_ns / baseline_ns,
            "delta_ns": magma_ns - baseline_ns,
        }
    )

rows.sort(key=lambda row: row["ratio"], reverse=True)
slowest = rows[:15]
fastest = sorted(rows, key=lambda row: row["ratio"])[:15]

ratios = sorted(row["ratio"] for row in rows)
median_ratio = ratios[len(ratios) // 2] if ratios else math.nan

lines = [
    "# MAGMA Decomposition Provider Comparison",
    "",
    f"- Baseline provider: `{baseline_path.name}`",
    f"- MAGMA provider: `{magma_path.name}`",
    f"- Common nabled decomposition cases: `{len(rows)}`",
]
if ratios:
    lines.append(f"- Median ratio (`magma/baseline`): `{median_ratio:.3f}`")
lines.append("")

lines.append("## Top Slowdowns (`magma/baseline`)")
lines.append("")
lines.append("| ratio | delta ns | domain | operation | dtype | size | benchmark |")
lines.append("|---:|---:|---|---|---|---:|---|")
for row in slowest:
    lines.append(
        f"| {row['ratio']:.3f} | {row['delta_ns']:.1f} | {row['domain']} | {row['operation']} | "
        f"{row['dtype']} | {row['size']} | `{row['full_id']}` |"
    )
lines.append("")

lines.append("## Top Speedups (`magma/baseline`)")
lines.append("")
lines.append("| ratio | delta ns | domain | operation | dtype | size | benchmark |")
lines.append("|---:|---:|---|---|---|---:|---|")
for row in fastest:
    lines.append(
        f"| {row['ratio']:.3f} | {row['delta_ns']:.1f} | {row['domain']} | {row['operation']} | "
        f"{row['dtype']} | {row['size']} | `{row['full_id']}` |"
    )
lines.append("")

report_path.write_text("\n".join(lines))
PY
}

for run in $(seq 1 "${REPEATS}"); do
  RUN_TAG="${BATCH_TS}-r${run}"
  BASELINE_JSON="${ARTIFACT_DIR}/${BASELINE_PROVIDER}-summary-${RUN_TAG}.json"
  MAGMA_JSON="${ARTIFACT_DIR}/${MAGMA_PROVIDER}-summary-${RUN_TAG}.json"
  REPORT_MD="${ARTIFACT_DIR}/comparison-${RUN_TAG}.md"

  if (( run % 2 == 1 )); then
    run_provider_pass "${BASELINE_PROVIDER_FEATURES}" "${BASELINE_JSON}"
    run_provider_pass "${MAGMA_PROVIDER_FEATURES}" "${MAGMA_JSON}"
  else
    run_provider_pass "${MAGMA_PROVIDER_FEATURES}" "${MAGMA_JSON}"
    run_provider_pass "${BASELINE_PROVIDER_FEATURES}" "${BASELINE_JSON}"
  fi
  compare_pass "${BASELINE_JSON}" "${MAGMA_JSON}" "${REPORT_MD}"
done

python3 - <<'PY' \
  "${ARTIFACT_DIR}" \
  "${BATCH_TS}" \
  "${REPEATS}" \
  "${BASELINE_PROVIDER}" \
  "${MAGMA_PROVIDER}" \
  "${BASELINE_PROVIDER_FEATURES}" \
  "${MAGMA_PROVIDER_FEATURES}" \
  "${PERSISTENT_RATIO_THRESHOLD}" \
  "${PERSISTENT_MIN_DELTA_NS}" \
  "${PERSISTENT_MIN_RUNS}" \
  "${ENFORCE_PERSISTENT_REGRESSIONS}"
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

artifact_dir = Path(sys.argv[1])
batch_ts = sys.argv[2]
repeats = int(sys.argv[3])
baseline_provider = sys.argv[4]
magma_provider = sys.argv[5]
baseline_provider_features = sys.argv[6]
magma_provider_features = sys.argv[7]
persistent_ratio_threshold = float(sys.argv[8])
persistent_min_delta_ns = float(sys.argv[9])
persistent_min_runs = int(sys.argv[10])
enforce_persistent = int(sys.argv[11]) != 0

domains = {
    "lu",
    "cholesky",
    "qr",
    "svd",
    "eigen",
    "schur",
    "polar",
    "sylvester",
    "matrix_functions",
}

ratios_per_case: dict[str, list[float]] = defaultdict(list)
deltas_per_case: dict[str, list[float]] = defaultdict(list)
meta: dict[str, tuple[str, str]] = {}
run_scope_stats: list[tuple[int, float, float]] = []

for run in range(1, repeats + 1):
    tag = f"{batch_ts}-r{run}"
    baseline_path = artifact_dir / f"{baseline_provider}-summary-{tag}.json"
    magma_path = artifact_dir / f"{magma_provider}-summary-{tag}.json"
    baseline_entries = json.loads(baseline_path.read_text())["entries"]
    magma_entries = json.loads(magma_path.read_text())["entries"]

    baseline_map = {
        entry["full_id"]: entry
        for entry in baseline_entries
        if entry.get("competitor") == "none" and "_nabled_" in entry.get("group_id", "")
    }
    magma_map = {
        entry["full_id"]: entry
        for entry in magma_entries
        if entry.get("competitor") == "none" and "_nabled_" in entry.get("group_id", "")
    }

    scope_ratios: list[float] = []
    for full_id, magma_entry in magma_map.items():
        baseline_entry = baseline_map.get(full_id)
        if baseline_entry is None:
            continue
        if magma_entry.get("domain") not in domains:
            continue
        baseline_ns = float(baseline_entry["mean_ns"])
        magma_ns = float(magma_entry["mean_ns"])
        if baseline_ns <= 0.0:
            continue
        ratio = magma_ns / baseline_ns
        scope_ratios.append(ratio)
        ratios_per_case[full_id].append(ratio)
        deltas_per_case[full_id].append(magma_ns - baseline_ns)
        meta[full_id] = (str(magma_entry.get("domain", "")), str(magma_entry.get("operation", "")))

    scope_ratios.sort()
    median = statistics.median(scope_ratios) if scope_ratios else float("nan")
    p90 = scope_ratios[int(0.9 * (len(scope_ratios) - 1))] if scope_ratios else float("nan")
    run_scope_stats.append((run, median, p90))

stable_rows = []
for full_id, values in ratios_per_case.items():
    deltas = deltas_per_case.get(full_id, [])
    if len(values) != repeats or len(deltas) != repeats:
        continue
    stable_rows.append(
        {
            "full_id": full_id,
            "domain": meta[full_id][0],
            "operation": meta[full_id][1],
            "ratios": values,
            "deltas_ns": deltas,
            "median_ratio": statistics.median(values),
            "mean_ratio": statistics.fmean(values),
            "min_ratio": min(values),
            "max_ratio": max(values),
            "median_delta_ns": statistics.median(deltas),
            "mean_delta_ns": statistics.fmean(deltas),
            "min_delta_ns": min(deltas),
            "max_delta_ns": max(deltas),
        }
    )

stable_rows.sort(key=lambda row: row["median_ratio"], reverse=True)

top_slow = stable_rows[:15]
top_fast = sorted(stable_rows, key=lambda row: row["median_ratio"])[:15]

if persistent_min_runs < 1:
    persistent_min_runs = 1
if persistent_min_runs > repeats:
    persistent_min_runs = repeats

persistent_regressions = []
for row in stable_rows:
    exceed_count = sum(
        1
        for ratio, delta_ns in zip(row["ratios"], row["deltas_ns"])
        if ratio > persistent_ratio_threshold and delta_ns > persistent_min_delta_ns
    )
    if exceed_count >= persistent_min_runs:
        persistent_regressions.append(
            {
                "full_id": row["full_id"],
                "domain": row["domain"],
                "operation": row["operation"],
                "ratios": row["ratios"],
                "median_ratio": row["median_ratio"],
                "min_ratio": row["min_ratio"],
                "max_ratio": row["max_ratio"],
                "median_delta_ns": row["median_delta_ns"],
                "min_delta_ns": row["min_delta_ns"],
                "max_delta_ns": row["max_delta_ns"],
                "exceed_count": exceed_count,
            }
        )

summary = {
    "batch_ts": batch_ts,
    "repeats": repeats,
    "baseline_provider": baseline_provider,
    "magma_provider": magma_provider,
    "baseline_provider_features": baseline_provider_features,
    "magma_provider_features": magma_provider_features,
    "persistent_ratio_threshold": persistent_ratio_threshold,
    "persistent_min_delta_ns": persistent_min_delta_ns,
    "persistent_min_runs": persistent_min_runs,
    "run_scope_stats": [
        {"run": run, "median_ratio": med, "p90_ratio": p90}
        for run, med, p90 in run_scope_stats
    ],
    "stable_case_count": len(stable_rows),
    "persistent_regression_count": len(persistent_regressions),
    "persistent_regressions": persistent_regressions,
    "top_slowdowns": top_slow,
    "top_speedups": top_fast,
}

json_path = artifact_dir / f"stability-{batch_ts}.json"
json_path.write_text(json.dumps(summary, indent=2))

lines = [
    "# MAGMA Decomposition Stability Summary",
    "",
    f"- Batch: `{batch_ts}`",
    f"- Repeats: `{repeats}`",
    f"- Stable decomposition cases across all repeats: `{len(stable_rows)}`",
    (
        f"- Persistent slowdown threshold: ratio > `{persistent_ratio_threshold:.3f}` "
        f"and delta > `{persistent_min_delta_ns:.1f}` ns in at least `{persistent_min_runs}` / `{repeats}` runs"
    ),
    f"- Persistent slowdowns: `{len(persistent_regressions)}`",
    "",
    "## Per-run Scope Ratios (`magma/openblas`)",
    "",
    "| run | median | p90 |",
    "|---:|---:|---:|",
]
for run, med, p90 in run_scope_stats:
    lines.append(f"| {run} | {med:.3f} | {p90:.3f} |")

lines.extend(
    [
        "",
        "## Persistent Slowdowns (threshold-gated)",
        "",
        "| exceeds | median ratio | median delta ns | min ratio | max ratio | domain | operation | benchmark |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
)
for row in persistent_regressions:
    lines.append(
        f"| {row['exceed_count']}/{repeats} | {row['median_ratio']:.3f} | {row['median_delta_ns']:.1f} | "
        f"{row['min_ratio']:.3f} | {row['max_ratio']:.3f} | {row['domain']} | {row['operation']} | `{row['full_id']}` |"
    )

lines.extend(
    [
        "",
        "## Persistent Slowdowns (by median ratio)",
        "",
        "| median | min | max | domain | operation | benchmark |",
        "|---:|---:|---:|---|---|---|",
    ]
)
for row in top_slow:
    lines.append(
        f"| {row['median_ratio']:.3f} | {row['min_ratio']:.3f} | {row['max_ratio']:.3f} | "
        f"{row['domain']} | {row['operation']} | `{row['full_id']}` |"
    )

lines.extend(
    [
        "",
        "## Persistent Speedups (by median ratio)",
        "",
        "| median | min | max | domain | operation | benchmark |",
        "|---:|---:|---:|---|---|---|",
    ]
)
for row in top_fast:
    lines.append(
        f"| {row['median_ratio']:.3f} | {row['min_ratio']:.3f} | {row['max_ratio']:.3f} | "
        f"{row['domain']} | {row['operation']} | `{row['full_id']}` |"
    )

md_path = artifact_dir / f"stability-{batch_ts}.md"
md_path.write_text("\n".join(lines) + "\n")

if enforce_persistent and persistent_regressions:
    print(
        "K-005 monitor failure: persistent MAGMA decomposition slowdowns detected "
        f"({len(persistent_regressions)} case(s)); see {md_path}",
        file=sys.stderr,
    )
    sys.exit(7)
PY

cp "${ARTIFACT_DIR}/comparison-${BATCH_TS}-r${REPEATS}.md" "${ARTIFACT_DIR}/comparison-latest.md"

echo "decomposition provider benchmark artifacts generated:"
echo "  bench command: ${BENCH_COMMAND}"
echo "  baseline provider label/features: ${BASELINE_PROVIDER} / ${BASELINE_PROVIDER_FEATURES}"
echo "  magma provider label/features: ${MAGMA_PROVIDER} / ${MAGMA_PROVIDER_FEATURES}"
echo "  ${ARTIFACT_DIR}/${BASELINE_PROVIDER}-summary-${BATCH_TS}-r1.json .. r${REPEATS}.json"
echo "  ${ARTIFACT_DIR}/${MAGMA_PROVIDER}-summary-${BATCH_TS}-r1.json .. r${REPEATS}.json"
echo "  ${ARTIFACT_DIR}/comparison-${BATCH_TS}-r1.md .. r${REPEATS}.md"
echo "  ${ARTIFACT_DIR}/stability-${BATCH_TS}.json"
echo "  ${ARTIFACT_DIR}/stability-${BATCH_TS}.md"
