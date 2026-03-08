# MAGMA Signoff Matrix

Last updated: 2026-03-07

## Purpose

This document is the release-blocking signoff ledger for MAGMA-integrated paths.

It tracks:

1. Which public APIs can execute through MAGMA.
2. Which feature-flag combinations permit MAGMA execution.
3. Which runtime policy gates may route away from MAGMA.
4. What evidence proves MAGMA execution and correctness.
5. Which rows are still pending for production-grade MAGMA confidence.

## Function-Level Expansion

Function-level one-row-per-public-function coverage is now tracked in:

1. `docs/MAGMA_PUBLIC_API_MATRIX.md`

That matrix references canonical route IDs from this file and is part of release-blocking MAGMA signoff.

## Profile Labels

1. `M`: `magma-system` enabled, `lapack-provider` disabled.
2. `ML`: both `magma-system` and `lapack-provider` enabled.
3. `M*`: any build with `magma-system` enabled (independent of `lapack-provider`).

## Evidence Artifacts

1. Execution-matrix tests:
   - `crates/nabled-linalg/src/magma_verification.rs`
   - `magma_verification::magma_dense_provider_execution_matrix`
   - `magma_verification::magma_sparse_provider_execution_matrix`
2. Remote logs:
   - `coverage/gpu-v2/magma/verification.log`
   - `coverage/gpu-v2/magma/strict-verification.log`
   - `coverage/gpu-v2/magma/strict-verification-20260307.log` (`rc=0`, script-refined strict run on RTX 4090)
   - `coverage/gpu-v2/magma/capability-batched-symbols-20260307.log` (remote symbol scan for batched decomposition feasibility)
   - `coverage/gpu-v2/magma/job-20260307T195643Z.log` (forced sparse strict matrix, `rc=0`)
   - `coverage/gpu-v2/magma/job-20260307T195705Z.log` (forced dense strict matrix, `rc=0`)
   - `coverage/gpu-v2/magma/job-20260307T195725Z.log` (`magma-strict-verify`, `rc=0`)
   - `coverage/gpu-v2/magma/job-20260307T203307Z.log` (`magma-strict-verify`, composed-domain rows included, `rc=0`)
   - `coverage/gpu-v2/magma/job-20260307T205521Z.log` (`magma-strict-verify`, script-refined baseline+forced-matrix flow, `rc=0`)
3. Strict mode contract:
   - `NABLED_MAGMA_STRICT=1` must fail fast (no silent provider fallback).

## Dense / Batched / Mixed Routes

| ID | Canonical API | Aliases / Derived APIs | Route Condition | Runtime Gate | Verification Evidence | Status |
|---|---|---|---|---|---|---|
| `MAG-D-001` | `lu::solve` | `lu::solve_view` | `M*` | `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `lu::solve` + provider/runtime call counters | Verified |
| `MAG-D-002` | `lu::inverse` | `lu::inverse_view` | `M*` | `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `lu::inverse` + counters | Verified |
| `MAG-D-003` | `lu::determinant` | `lu::determinant_view`, `lu::log_determinant`, `lu::log_determinant_view` | `M*` | `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `lu::determinant` + counters | Verified (`log_*` inherited) |
| `MAG-D-004` | `lu::solve_complex` | `lu::solve_complex_view` | `M*` | `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `lu::solve_complex` + counters | Verified |
| `MAG-D-005` | `lu::inverse_complex` | `lu::inverse_complex_view` | `M*` | `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `lu::inverse_complex` + counters | Verified |
| `MAG-D-006` | `lu::determinant_complex` | `lu::determinant_complex_view` | `M*` | `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `lu::determinant_complex` + counters | Verified |
| `MAG-D-007` | `lu::solve_mixed_f64` | `lu::solve_mixed_f64_view` | `M*` | none | dense matrix case `lu::solve_mixed_f64` + counters | Verified |
| `MAG-D-008` | `lu::solve_mixed_complex` | `lu::solve_mixed_complex_view` | `M*` | none | dense matrix case `lu::solve_mixed_complex` + counters | Verified |
| `MAG-D-009` | `cholesky::decompose` | `cholesky::decompose_view` | `M*` | none | dense matrix case `cholesky::decompose` + counters | Verified |
| `MAG-D-010` | `cholesky::solve` | `cholesky::solve_view`, `cholesky::solve_into` | `M*` | none | dense matrix case `cholesky::solve` + counters | Verified (`solve_into` inherited) |
| `MAG-D-011` | `cholesky::inverse` | `cholesky::inverse_view` | `M*` | none | dense matrix case `cholesky::inverse` + counters | Verified |
| `MAG-D-012` | `cholesky::decompose_complex` | `cholesky::decompose_complex_view` | `M*` | none | dense matrix case `cholesky::decompose_complex` + counters | Verified |
| `MAG-D-013` | `cholesky::solve_complex` | `cholesky::solve_complex_view` | `M*` | none | dense matrix case `cholesky::solve_complex` + counters | Verified |
| `MAG-D-014` | `cholesky::inverse_complex` | `cholesky::inverse_complex_view` | `M*` | none | dense matrix case `cholesky::inverse_complex` + counters | Verified |
| `MAG-D-015` | `qr::decompose` | `qr::decompose_view`, `qr::decompose_reduced`, `qr::decompose_with_pivoting` | `M*` | `nrows >= ncols` and `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `qr::decompose` + counters | Verified (`reduced/pivoted` inherited) |
| `MAG-D-016` | `qr::decompose_complex` | `qr::decompose_complex_view` | `M*` | `nrows >= ncols` and `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `qr::decompose_complex` + counters | Verified |
| `MAG-D-017` | `qr::solve_least_squares` | `qr::solve_least_squares_view` | `M*` | `M`: inherits real `qr::decompose` gates; `ML`: MAGMA-first under real `qr::decompose` gates with lapack fallback | dense matrix case `qr::solve_least_squares` + counters | Verified |
| `MAG-D-018` | `svd::decompose` | `svd::decompose_view`, `svd::decompose_with_tolerance`, `svd::decompose_truncated`, `svd::pseudo_inverse`, `svd::pseudo_inverse_into`, `svd::null_space` | `M` only | `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `svd::decompose` + counters | Verified (`derived` inherited) |
| `MAG-D-019` | `svd::decompose_complex` | `svd::decompose_complex_view` | `M*` | `DenseKernelPolicy::prefer_magma_decomposition` | dense matrix case `svd::decompose_complex` + counters | Verified |
| `MAG-D-020` | `eigen::symmetric` | `eigen::symmetric_view` | `M` only | none | dense matrix case `eigen::symmetric` + counters | Verified |
| `MAG-D-021` | `eigen::generalized` | `eigen::generalized_view` | `M*` | none | dense matrix case `eigen::generalized` + counters | Verified |
| `MAG-D-022` | `eigen::nonsymmetric_complex` | `eigen::nonsymmetric_complex_view` | `M*` | none | dense matrix case `eigen::nonsymmetric_complex` + counters | Verified |
| `MAG-D-023` | `batched::lu` | `batched::lu_view` | `M*` | `DenseKernelPolicy::prefer_magma_batched_decomposition` | dense matrix case `batched::lu` + counters | Verified |
| `MAG-D-024` | `batched::cholesky` | `batched::cholesky_view` | `M*` | `DenseKernelPolicy::prefer_magma_batched_decomposition` | dense matrix case `batched::cholesky` + counters | Verified |
| `MAG-D-025` | `batched::qr` | `batched::qr_view` | `M*` | `DenseKernelPolicy::prefer_magma_batched_decomposition` | dense matrix case `batched::qr` + counters | Verified |
| `MAG-D-026` | `batched::svd` | `batched::svd_view` | `M*` | `DenseKernelPolicy::prefer_magma_batched_decomposition` (attempt MAGMA per-slice route first, fallback per strict policy) | dense matrix case `batched::svd` + counters | Verified |
| `MAG-D-027` | `batched::symmetric_eigen` | `batched::symmetric_eigen_view` | `M*` | `rows == cols` and `DenseKernelPolicy::prefer_magma_batched_decomposition` (attempt MAGMA per-slice route first, fallback per strict policy) | dense matrix case `batched::symmetric_eigen` + counters | Verified |
| `MAG-D-028` | `sylvester::solve_sylvester_mixed_f64` | `sylvester::solve_sylvester_mixed_f64_view`, `sylvester::solve_lyapunov_mixed_f64`, `sylvester::solve_lyapunov_mixed_f64_view` | `M*` | none | dense matrix case `solve_sylvester_mixed_f64` and `solve_lyapunov_mixed_f64` + counters | Verified |
| `MAG-D-029` | `sylvester::solve_sylvester_mixed_complex` | `sylvester::solve_sylvester_mixed_complex_view`, `sylvester::solve_lyapunov_mixed_complex`, `sylvester::solve_lyapunov_mixed_complex_view` | `M*` | none | dense matrix case `solve_sylvester_mixed_complex` and `solve_lyapunov_mixed_complex` + counters | Verified |
| `MAG-D-030` | `schur::compute_schur` | `schur::compute_schur_view`, `schur::compute_schur_into`, `schur::compute_schur_into_view`, `schur::compute_schur_with_workspace_into` | `M*` | inherits `qr::decompose` gates (`nrows >= ncols`, `DenseKernelPolicy::prefer_magma_decomposition`) | dense matrix case `schur::compute_schur` + counters | Verified |
| `MAG-D-031` | `schur::compute_schur_complex` | `schur::compute_schur_complex_view`, `schur::compute_schur_complex_into`, `schur::compute_schur_complex_into_view`, `schur::compute_schur_complex_with_workspace_into` | `M*` | inherits `qr::decompose_complex` gates (`nrows >= ncols`, `DenseKernelPolicy::prefer_magma_decomposition`) | dense matrix case `schur::compute_schur_complex` + counters | Verified |
| `MAG-D-032` | `polar::compute_polar` | `polar::compute_polar_view` | `M` only | inherits real `svd::decompose_view` gate (`DenseKernelPolicy::prefer_magma_decomposition`) in non-LAPACK path | dense matrix case `polar::compute_polar` + counters | Verified |
| `MAG-D-033` | `polar::compute_polar_complex` | `polar::compute_polar_complex_view` | `M*` | `M`: inherits iterative `lu::inverse_complex` decomposition gate; `ML`: inherits `svd::decompose_complex_view` gate | dense matrix case `polar::compute_polar_complex` + counters | Verified |
| `MAG-D-034` | `matrix_functions::matrix_exp_eigen` | `matrix_functions::matrix_exp_eigen_view` | `M` only | symmetric-input gate + inherited `eigen::symmetric_view` provider route | dense matrix case `matrix_functions::matrix_exp_eigen` + counters | Verified |
| `MAG-D-035` | `matrix_functions::matrix_exp_eigen_complex` | `matrix_functions::matrix_exp_eigen_complex_view` | `M` only | Hermitian-input gate + inherited `hermitian_eigen_dispatch` (Schur-complex in non-LAPACK path) | dense matrix case `matrix_functions::matrix_exp_eigen_complex` + counters | Verified |
| `MAG-D-036` | `matrix_functions::matrix_log_eigen` | `matrix_functions::matrix_log_eigen_view`, `matrix_functions::matrix_log_eigen_into`, `matrix_functions::matrix_log_eigen_with_workspace_into` | `M` only | symmetric positive-definite gate + inherited `eigen::symmetric_view` provider route | dense matrix case `matrix_functions::matrix_log_eigen` + counters | Verified |
| `MAG-D-037` | `matrix_functions::matrix_log_eigen_complex` | `matrix_functions::matrix_log_eigen_complex_view`, `matrix_functions::matrix_log_eigen_complex_into`, `matrix_functions::matrix_log_eigen_complex_with_workspace_into` | `M` only | Hermitian positive-definite gate + inherited `hermitian_eigen_dispatch` (Schur-complex in non-LAPACK path) | dense matrix case `matrix_functions::matrix_log_eigen_complex` + counters | Verified |
| `MAG-D-038` | `matrix_functions::matrix_log_svd` | `matrix_functions::matrix_log_svd_view`, `matrix_functions::matrix_log_svd_into`, `matrix_functions::matrix_log_svd_with_workspace_into` | `M` only | square + positive singular-values gate + inherited real `svd::decompose_view` route | dense matrix case `matrix_functions::matrix_log_svd` + counters | Verified |
| `MAG-D-039` | `matrix_functions::matrix_log_svd_complex` | `matrix_functions::matrix_log_svd_complex_view`, `matrix_functions::matrix_log_svd_complex_into`, `matrix_functions::matrix_log_svd_complex_with_workspace_into` | `M*` | square + positive singular-values gate + inherited complex `svd::decompose_complex_view` route | dense matrix case `matrix_functions::matrix_log_svd_complex` + counters | Verified |
| `MAG-D-040` | `matrix_functions::matrix_power` | `matrix_functions::matrix_power_view`, `matrix_functions::matrix_power_into`, `matrix_functions::matrix_power_with_workspace_into` | `M` only | symmetric-input gate + inherited `eigen::symmetric_view` provider route | dense matrix case `matrix_functions::matrix_power` + counters | Verified |
| `MAG-D-041` | `matrix_functions::matrix_power_complex` | `matrix_functions::matrix_power_complex_view`, `matrix_functions::matrix_power_complex_into`, `matrix_functions::matrix_power_complex_with_workspace_into` | `M` only | Hermitian-input gate + inherited `hermitian_eigen_dispatch` (Schur-complex in non-LAPACK path) | dense matrix case `matrix_functions::matrix_power_complex` + counters | Verified |
| `MAG-D-042` | `matrix_functions::matrix_sign` | `matrix_functions::matrix_sign_view`, `matrix_functions::matrix_sign_into`, `matrix_functions::matrix_sign_with_workspace_into` | `M` only | symmetric-input gate + inherited `eigen::symmetric_view` provider route | dense matrix case `matrix_functions::matrix_sign` + counters | Verified |
| `MAG-D-043` | `matrix_functions::matrix_sign_complex` | `matrix_functions::matrix_sign_complex_view`, `matrix_functions::matrix_sign_complex_into`, `matrix_functions::matrix_sign_complex_with_workspace_into` | `M` only | Hermitian-input gate + inherited `hermitian_eigen_dispatch` (Schur-complex in non-LAPACK path) | dense matrix case `matrix_functions::matrix_sign_complex` + counters | Verified |

## Sparse MAGMA Routes

| ID | Canonical API | Aliases / Derived APIs | Route Condition | Runtime Gate | Verification Evidence | Status |
|---|---|---|---|---|---|---|
| `MAG-S-001` | `sparse::matvec_magma_f64_view` | `sparse::matvec_magma_f64_view_into` | `M*` | none | sparse matrix case `matvec_magma_f64_view` + sparse provider counter | Verified (`_into` inherited) |
| `MAG-S-002` | `sparse::matvec_magma_f32_view` | `sparse::matvec_magma_f32_view_into` | `M*` | none | sparse matrix case `matvec_magma_f32_view` + sparse provider counter | Verified (`_into` inherited) |
| `MAG-S-003` | `sparse::matmat_dense_magma_f64_view` | `sparse::matmat_dense_magma_f64_view_into` | `M*` | size floor (`>=16`) unless verify-force mode | sparse matrix case `matmat_dense_magma_f64_view` + sparse provider counter | Verified (`_into` inherited) |
| `MAG-S-004` | `sparse::matmat_dense_magma_f32_view` | `sparse::matmat_dense_magma_f32_view_into` | `M*` | size floor (`>=16`) unless verify-force mode | sparse matrix case `matmat_dense_magma_f32_view` + sparse provider counter | Verified (`_into` inherited) |
| `MAG-S-005` | `sparse::conjugate_gradient_magma_f64_view` | none | `M*` | none | sparse matrix case `conjugate_gradient_magma_f64_view` + sparse provider counter | Verified |
| `MAG-S-006` | `sparse::conjugate_gradient_magma_f32_view` | none | `M*` | none | sparse matrix case `conjugate_gradient_magma_f32_view` + sparse provider counter | Verified |
| `MAG-S-007` | `sparse::pcg_jacobi_magma_f64_view` | none | `M*` | none | sparse matrix case `pcg_jacobi_magma_f64_view` + sparse provider counter | Verified |
| `MAG-S-008` | `sparse::pcg_jacobi_magma_f32_view` | none | `M*` | none | sparse matrix case `pcg_jacobi_magma_f32_view` + sparse provider counter | Verified |
| `MAG-S-009` | `sparse::gmres_magma_f64_view` | none | `M*` | none | sparse matrix case `gmres_magma_f64_view` + sparse provider counter | Verified |
| `MAG-S-010` | `sparse::gmres_magma_f32_view` | none | `M*` | none | sparse matrix case `gmres_magma_f32_view` + sparse provider counter | Verified |
| `MAG-S-011` | `sparse::gmres_ilu0_magma_f64_view` | none | `M*` | none | sparse matrix case `gmres_ilu0_magma_f64_view` + sparse provider counter | Verified |
| `MAG-S-012` | `sparse::gmres_ilu0_magma_f32_view` | none | `M*` | none | sparse matrix case `gmres_ilu0_magma_f32_view` + sparse provider counter | Verified |
| `MAG-S-013` | `sparse::bicgstab_magma_f64_view` | none | `M*` | none | sparse matrix case `bicgstab_magma_f64_view` + sparse provider counter | Verified |
| `MAG-S-014` | `sparse::bicgstab_magma_f32_view` | none | `M*` | none | sparse matrix case `bicgstab_magma_f32_view` + sparse provider counter | Verified |
| `MAG-S-015` | `sparse::bicgstab_ilu0_magma_f64_view` | none | `M*` | none | sparse matrix case `bicgstab_ilu0_magma_f64_view` + sparse provider counter | Verified |
| `MAG-S-016` | `sparse::bicgstab_ilu0_magma_f32_view` | none | `M*` | none | sparse matrix case `bicgstab_ilu0_magma_f32_view` + sparse provider counter | Verified |

## Current Production-Grade Gaps

All currently routed MAGMA API rows in this matrix now have direct execution proof.

`MAG-L-004` is now closed: forced strict verification runs are clean (no `cusparseCreate` /
`cusparseSetStream` context-noise lines), and the strict job now separates baseline correctness
(`NABLED_MAGMA_STRICT=0`) from forced execution-matrix strict routing to avoid false failures.

Remaining MAG-L backlog is closed for the currently scoped MAGMA surface:

1. `MAG-L-001`: complete. `batched::svd*` now attempts MAGMA route in `M*` builds via batched policy gating; native MAGMA batched SVD symbols were scanned and are absent in the current runtime (`capability-batched-symbols-20260307.log`), so per-slice MAGMA routing is the explicit contract.
2. `MAG-L-002`: complete. `batched::symmetric_eigen*` now attempts MAGMA route in `M*` builds via batched policy gating; native MAGMA batched symmetric-eigen symbols were scanned and are absent in the current runtime (`capability-batched-symbols-20260307.log`), so per-slice MAGMA routing is the explicit contract.
3. `MAG-L-003`: complete. Composed-domain signoff (`schur`, `polar`, matrix-functions) is now covered by routed execution matrix rows `MAG-D-030..MAG-D-043` with strict verification evidence (`job-20260307T203307Z.log`, `rc=0`).
4. `MAG-L-004`: complete. Forced strict verification runs are clean (no `cusparseCreate` / `cusparseSetStream` context-noise lines).
5. `MAG-L-005`: complete (expanded to one row per MAGMA-scope public function in `docs/MAGMA_PUBLIC_API_MATRIX.md`).

## Update Rules

1. Do not mark a row `Verified` without:
   - explicit execution proof (provider call counters and/or strict-path evidence),
   - correctness assertion coverage,
   - strict-mode validation (`NABLED_MAGMA_STRICT=1`) where applicable.
2. If a public API is added that can route to MAGMA, add a new row in this file in the same change set.
3. Keep `docs/GPU_V2_TRACKER.md`, `docs/STATUS.md`, and this file synchronized when status changes.
