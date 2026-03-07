#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

BASELINE_PROVIDER="${BASELINE_PROVIDER:-openblas-system}"
MAGMA_PROVIDER="${MAGMA_PROVIDER:-magma-system}"
ARTIFACT_DIR="${ARTIFACT_DIR:-coverage/gpu-v2/magma/bench}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${ARTIFACT_DIR}"

run_provider_pass() {
  local provider="$1"
  local out_json="$2"

  rm -rf target/criterion crates/nabled/target/criterion
  NABLED_PROVIDER_BENCH_FEATURES="${provider}" just -f .justfile bench-smoke-report-provider
  cp coverage/benchmarks/summary.json "${out_json}"
}

BASELINE_JSON="${ARTIFACT_DIR}/${BASELINE_PROVIDER}-summary-${TS}.json"
MAGMA_JSON="${ARTIFACT_DIR}/${MAGMA_PROVIDER}-summary-${TS}.json"
REPORT_MD="${ARTIFACT_DIR}/comparison-${TS}.md"
LATEST_REPORT_MD="${ARTIFACT_DIR}/comparison-latest.md"

run_provider_pass "${BASELINE_PROVIDER}" "${BASELINE_JSON}"
run_provider_pass "${MAGMA_PROVIDER}" "${MAGMA_JSON}"

cp "${BASELINE_JSON}" "${ARTIFACT_DIR}/${BASELINE_PROVIDER}-summary.json"
cp "${MAGMA_JSON}" "${ARTIFACT_DIR}/${MAGMA_PROVIDER}-summary.json"

python3 - <<'PY' "${BASELINE_JSON}" "${MAGMA_JSON}" "${REPORT_MD}"
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
    if entry.get("competitor") == "none"
}
magma_map = {
    entry["full_id"]: entry
    for entry in magma
    if entry.get("competitor") == "none"
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
    ratio = magma_ns / baseline_ns
    rows.append(
        {
            "full_id": full_id,
            "domain": magma_entry.get("domain", ""),
            "operation": magma_entry.get("operation", ""),
            "dtype": magma_entry.get("dtype", ""),
            "size": magma_entry.get("size", ""),
            "baseline_ns": baseline_ns,
            "magma_ns": magma_ns,
            "ratio": ratio,
            "delta_ns": magma_ns - baseline_ns,
        }
    )

rows.sort(key=lambda row: row["ratio"], reverse=True)
slowest = rows[:25]
fastest = sorted(rows, key=lambda row: row["ratio"])[:25]

ratios = [row["ratio"] for row in rows]
median_ratio = sorted(ratios)[len(ratios) // 2] if ratios else math.nan

lines = []
lines.append("# MAGMA Provider Benchmark Comparison")
lines.append("")
lines.append(f"- Baseline provider: `{baseline_path.name}`")
lines.append(f"- MAGMA provider: `{magma_path.name}`")
lines.append(f"- Common nabled cases: `{len(rows)}`")
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

cp "${REPORT_MD}" "${LATEST_REPORT_MD}"

echo "provider benchmark artifacts generated:"
echo "  ${BASELINE_JSON}"
echo "  ${MAGMA_JSON}"
echo "  ${REPORT_MD}"
echo "  ${LATEST_REPORT_MD}"
