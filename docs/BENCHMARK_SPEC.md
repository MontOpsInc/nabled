# Benchmark Specification

This document defines how performance and numerical accuracy benchmarks are run and reported for nabled.

## Goals

1. Track performance regressions before release.
2. Compare nabled against ecosystem baselines.
3. Require correctness checks alongside timing.

## Initial Scope (Phase 1 Pilot)

Pilot domains:

1. SVD
2. QR

Pilot backends:

1. nalgebra path in nabled
2. ndarray path in nabled

Target competitor set for pilot comparisons:

1. `nalgebra`
2. `faer`
3. `ndarray-linalg`
4. `nalgebra-lapack`

Current implemented competitor harnesses:

1. direct `nalgebra` for SVD and QR
2. direct `faer` for SVD and QR
3. `nalgebra-lapack` for SVD and QR (enabled with `lapack-competitors` feature, Linux target)
4. `ndarray-linalg` for SVD and QR (enabled with `lapack-competitors` feature, Linux target)

## Workload Matrix

Each benchmark suite should cover:

1. Sizes:
   - small: `16`, `32`, `64`
   - medium: `128`, `256`
   - large: `512`, `1024` (nightly/scheduled jobs if too slow for PR CI)
2. Shapes:
   - square: `n x n`
   - tall-skinny: `2n x n`
   - wide-short: `n x 2n`
3. Scalar types:
   - `f32`
   - `f64`
4. Input classes:
   - well-conditioned random dense
   - rank-deficient synthetic cases
   - near-singular stress cases

## Correctness Contracts

Timing results are valid only if correctness checks pass.

SVD checks:

1. Reconstruction error: `||A - U S V^T||_F / ||A||_F <= tol`.
2. Singular values non-negative and sorted (within tolerance).
3. Rank-sensitive cases produce expected null/rank behavior.

QR checks:

1. Reconstruction error: `||A - Q R||_F / ||A||_F <= tol` (or `Q R P^T` for pivoted).
2. Orthogonality error: `||Q^T Q - I||_F <= tol`.
3. Least-squares residual meets expected bound versus baseline.

Tolerance defaults:

1. `f64`: `1e-10`
2. `f32`: `1e-5`

## Reporting Format

Store outputs under `coverage/benchmarks/`:

1. `summary.json`:
   - benchmark metadata (`git_sha`, `rustc`, `date`, `cpu`)
   - per-case median/mean/stddev
   - correctness pass/fail
2. `summary.csv`:
   - one row per benchmark case
   - columns: `domain,backend,competitor,size,shape,dtype,time_ns,throughput,correctness`
3. `regressions.md`:
   - human-readable delta vs baseline branch/tag
   - flagged cases above regression threshold

Current commands:

1. `just bench-smoke` runs quick SVD + QR benchmark suites.
2. `just bench-report` parses Criterion output and writes `summary.json`, `summary.csv`, `regressions.md`.
3. `just bench-smoke-report` runs both in sequence.
4. `just bench-report-check` runs report generation and fails if baseline regressions exceed the `>10%` threshold.
5. `just bench-smoke-check` runs smoke benches plus regression enforcement in one command.
6. `just bench-baseline-update` promotes the latest `summary.json` to `baseline/summary.json`.
7. `just bench-smoke-lapack` runs smoke SVD + QR suites with `lapack-competitors` enabled.
8. `just bench-smoke-report-lapack` runs lapack-enabled smoke suites and regenerates report artifacts.
9. LAPACK smoke commands require Linux with system BLAS/LAPACK libraries available (`libopenblas-dev`, `liblapack-dev`).

## CI Policy

1. PR CI runs a smoke subset (small + medium sizes).
2. Scheduled CI runs the full matrix.
3. Fail conditions:
   - any correctness failure
   - performance regression above threshold on protected cases
4. Regression enforcement requires a baseline file at `coverage/benchmarks/baseline/summary.json`.
5. Protected regression cases are nabled-owned paths only (`competitor == none`), not external competitor timings.

Initial regression threshold:

1. warning at `>5%`
2. fail at `>10%` on protected pilot cases

## Baseline Management

1. Release-quality baselines should be pinned to a tagged commit or explicit SHA.
2. Benchmark environment metadata must be recorded with every run.
3. Update baseline only via explicit maintainers’ approval.
4. The report tool reads baseline from `coverage/benchmarks/baseline/summary.json`.
5. CI can restore this baseline path from branch-local cache before checks and update it after a successful run.
