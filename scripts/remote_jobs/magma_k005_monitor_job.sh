#!/usr/bin/env bash
set -euo pipefail

# K-005 monitor mode:
# - repeats provider decomposition compare on the same host
# - fails only on persistent regressions (not single-run variance)

REPEATS="${REPEATS:-5}"
PERSISTENT_RATIO_THRESHOLD="${PERSISTENT_RATIO_THRESHOLD:-1.03}"
PERSISTENT_MIN_RUNS="${PERSISTENT_MIN_RUNS:-4}"
ENFORCE_PERSISTENT_REGRESSIONS="${ENFORCE_PERSISTENT_REGRESSIONS:-1}"

export REPEATS
export PERSISTENT_RATIO_THRESHOLD
export PERSISTENT_MIN_RUNS
export ENFORCE_PERSISTENT_REGRESSIONS

bash scripts/remote_jobs/magma_provider_bench_decomposition_job.sh
