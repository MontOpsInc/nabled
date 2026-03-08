# Reference Rubric

Last updated: 2026-03-08

## Purpose

This document defines how `nabled` decides that a capability area is "done"
and when the library can be called "production-ready v1."

The goal is objective convergence, not open-ended feature growth.

## Why Multi-Reference (Not Single-Library)

No single external library is a sufficient anchor across all of `nabled` domains.

1. `SciPy`/`NumPy` are strong anchors for dense/sparse linear algebra and many matrix functions.
2. `TensorLy` is a stronger anchor for tensor decomposition/tensor-network workflows (CP/Tucker/TT).
3. `scikit-learn` is a stronger anchor for PCA/regression/statistical ML primitives.
4. `CuPy`/`PyTorch` are useful anchors for GPU kernel behavior/performance sanity checks.

This is a rubric anchor only. `nabled` does not depend on these libraries.

## Domain Anchor Map

| Domain | Primary external anchor | Scope |
|---|---|---|
| Dense decompositions/functions | NumPy/SciPy | QR/SVD/LU/Cholesky/Eigen/Schur, matrix functions |
| Sparse kernels/solvers | SciPy sparse/sparse.linalg | sparse matvec/matmat, direct and iterative sparse solves |
| Tensor decomposition/network | TensorLy | CP/Tucker/TT decomposition + algebra/ergonomics |
| ML/stat primitives | scikit-learn (and SciPy where needed) | PCA/regression/statistics-facing numerics |
| GPU kernel behavior | CuPy/PyTorch (sanity/perf), internal GPU signoff | backend parity and performance envelope checks |

## Done Criteria (Per Domain)

A domain is "done" only when all gates below are true:

1. Capability parity map is explicit.
   - Every targeted external-anchor operation is mapped to a `nabled` public API.
   - Any intentionally unsupported anchor operation is explicitly documented.
2. Correctness parity passes.
   - Deterministic fixture tests.
   - Randomized/property tests where applicable.
   - Numeric thresholds are explicit for `f64` and `f32`.
3. Feature/build matrix coverage passes.
   - Required permutations run and are green:
     - internal (`no-default-features` where applicable),
     - provider-enabled (`lapack-provider`/provider features),
     - GPU-enabled (`accelerator-wgpu`/`magma-system`) where applicable.
4. Performance envelope is acceptable.
   - Benchmark chunk exists and is reproducible.
   - No persistent severe regressions against chosen anchors for in-scope workloads.
5. API/contract quality is aligned.
   - No hidden copy-heavy behavior in hot paths.
   - `*_view`/`*_into` and workspace semantics are explicit where relevant.
   - Error semantics are typed and documented.
6. Documentation and tracker state are synchronized.
   - `CAPABILITY_MATRIX`, `STATUS`, and `EXECUTION_TRACKER` reflect the landed state.

## Production-Ready v1 Definition

`nabled` is production-ready v1 when:

1. All required v1 gates in `docs/EXECUTION_TRACKER.md` are complete.
2. Every in-scope domain in `docs/CAPABILITY_MATRIX.md` satisfies the done criteria above.
3. Quality gates pass consistently (`just checks`, clippy/test matrices, coverage policy).
4. No-surprises audit is clean (allocation contracts, fallback/provider behavior, docs parity).

## Post-v1 Tensor Bound (Finite)

Tensor expansion beyond the locked v1 surface is bounded by this explicit sequence:

1. `D-179`: TT algebra utilities (`tt_inner`, `tt_norm`, `tt_add`, `tt_hadamard`, `tt_hadamard_round`). ✅
2. `D-181`: CP diagnostics (`fit`, residual/error metrics, convergence-report helpers). ✅
3. `D-182`: Tucker ergonomics/utilities (core transforms/projections and bounded convenience helpers). ✅
4. After `D-182`, tensor-depth expansion returns to monitor mode unless a new tracker item is approved.

## Governance

If scope pressure appears ("keep adding algorithms"), this rubric wins.

1. Additions outside the mapped anchor scope require a new tracker ID and rationale.
2. Existing done items are not reopened unless correctness/performance regressions are observed.
