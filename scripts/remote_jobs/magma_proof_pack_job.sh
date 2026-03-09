#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

ARTIFACT_DIR="${ARTIFACT_DIR:-coverage/gpu-v2/magma/bench/decomposition}"
REPEATS="${REPEATS:-3}"
BATCH_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PROOF_MD="${ARTIFACT_DIR}/proof-pack-${BATCH_TS}.md"

export BENCH_COMMAND="${BENCH_COMMAND:-bench-smoke-report-provider-decomposition-lto}"
export REPEATS
export ENFORCE_PERSISTENT_REGRESSIONS="${ENFORCE_PERSISTENT_REGRESSIONS:-0}"
export PERSISTENT_RATIO_THRESHOLD="${PERSISTENT_RATIO_THRESHOLD:-1.03}"
export PERSISTENT_MIN_DELTA_NS="${PERSISTENT_MIN_DELTA_NS:-5000}"
export PERSISTENT_MIN_RUNS="${PERSISTENT_MIN_RUNS:-${REPEATS}}"

bash scripts/remote_jobs/magma_provider_bench_decomposition_job.sh

LATEST_STABILITY="$(ls -t "${ARTIFACT_DIR}"/stability-*.json 2>/dev/null | head -n1 || true)"
LATEST_COMPARE="$(ls -t "${ARTIFACT_DIR}"/comparison-*.md 2>/dev/null | head -n1 || true)"

if [[ -z "${LATEST_STABILITY}" ]]; then
  echo "No stability json found in ${ARTIFACT_DIR}" >&2
  exit 1
fi

python3 - <<'PY' "${LATEST_STABILITY}" "${LATEST_COMPARE}" "${PROOF_MD}" "${BENCH_COMMAND}"
from __future__ import annotations

import json
import sys
from pathlib import Path

stability_path = Path(sys.argv[1])
comparison_path = Path(sys.argv[2]) if sys.argv[2] else None
proof_path = Path(sys.argv[3])
bench_command = sys.argv[4]

data = json.loads(stability_path.read_text())
run_stats = data.get("run_scope_stats", [])
top_slow = data.get("top_slowdowns", [])[:5]
top_fast = data.get("top_speedups", [])[:5]
persistent = data.get("persistent_regressions", [])[:5]
baseline_provider = data.get("baseline_provider", "openblas-system")
magma_provider = data.get("magma_provider", "magma-system")
baseline_features = data.get("baseline_provider_features", baseline_provider)
magma_features = data.get("magma_provider_features", magma_provider)

lines = [
    "# MAGMA Proof Pack (LTO Decomposition Scope)",
    "",
    f"- Source stability artifact: `{stability_path.name}`",
    f"- Source comparison artifact: `{comparison_path.name if comparison_path else 'n/a'}`",
    f"- Baseline provider label/features: `{baseline_provider}` / `{baseline_features}`",
    f"- MAGMA provider label/features: `{magma_provider}` / `{magma_features}`",
    f"- Bench command: `{bench_command}`",
    f"- Repeats: `{data.get('repeats', 'n/a')}`",
    f"- Persistent threshold: ratio > `{data.get('persistent_ratio_threshold', 'n/a')}` "
    f"and delta > `{data.get('persistent_min_delta_ns', 'n/a')}` ns "
    f"in at least `{data.get('persistent_min_runs', 'n/a')}` run(s)",
    "",
    f"## Run-level scope stats (`{magma_provider}/{baseline_provider}`)",
    "",
    "| run | median | p90 |",
    "|---:|---:|---:|",
]

for row in run_stats:
    lines.append(
        f"| {row.get('run', 'n/a')} | {float(row.get('median_ratio', 0.0)):.3f} | "
        f"{float(row.get('p90_ratio', 0.0)):.3f} |"
    )

lines.extend(
    [
        "",
        "## Strongest speedups",
        "",
        "| median ratio | domain | operation | benchmark |",
        "|---:|---|---|---|",
    ]
)
for row in top_fast:
    lines.append(
        f"| {float(row.get('median_ratio', 0.0)):.3f} | {row.get('domain', '')} | "
        f"{row.get('operation', '')} | `{row.get('full_id', '')}` |"
    )

lines.extend(
    [
        "",
        "## Strongest slowdowns",
        "",
        "| median ratio | domain | operation | benchmark |",
        "|---:|---|---|---|",
    ]
)
for row in top_slow:
    lines.append(
        f"| {float(row.get('median_ratio', 0.0)):.3f} | {row.get('domain', '')} | "
        f"{row.get('operation', '')} | `{row.get('full_id', '')}` |"
    )

lines.extend(
    [
        "",
        "## Persistent slowdowns (threshold-gated)",
        "",
        "| exceed count | median ratio | median delta ns | domain | operation | benchmark |",
        "|---:|---:|---:|---|---|---|",
    ]
)
if persistent:
    repeats = int(data.get("repeats", 1))
    for row in persistent:
        lines.append(
            f"| {int(row.get('exceed_count', 0))}/{repeats} | "
            f"{float(row.get('median_ratio', 0.0)):.3f} | {float(row.get('median_delta_ns', 0.0)):.1f} | {row.get('domain', '')} | "
            f"{row.get('operation', '')} | `{row.get('full_id', '')}` |"
        )
else:
    lines.append("| 0 | n/a | n/a | n/a | n/a | n/a |")

proof_path.write_text("\n".join(lines) + "\n")
PY

cp "${PROOF_MD}" "${ARTIFACT_DIR}/proof-pack-latest.md"
echo "proof pack generated:"
echo "  ${PROOF_MD}"
echo "  ${ARTIFACT_DIR}/proof-pack-latest.md"
