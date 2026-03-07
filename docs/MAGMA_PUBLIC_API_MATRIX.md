# MAGMA Public API Matrix

Last updated: 2026-03-07

## Purpose

This document expands MAGMA signoff coverage to one row per public function in MAGMA scope.

1. Every row is a concrete public function symbol.
2. `Route ID` references the canonical route rows in `docs/MAGMA_SIGNOFF.md`.
3. Profile/runtime-gate/evidence are inherited from the mapped route row.

## Conventions

1. `Status: Verified` means function-level coverage is explicitly mapped and inherits a verified route row.
2. Function names are listed once per public symbol (cfg-split internal/provider arms are represented by one public symbol row).

## LU

| Function | Route ID | Status |
|---|---|---|
| `lu::solve` | `MAG-D-001` | Verified |
| `lu::solve_view` | `MAG-D-001` | Verified |
| `lu::inverse` | `MAG-D-002` | Verified |
| `lu::inverse_view` | `MAG-D-002` | Verified |
| `lu::determinant` | `MAG-D-003` | Verified |
| `lu::determinant_view` | `MAG-D-003` | Verified |
| `lu::log_determinant` | `MAG-D-003` | Verified |
| `lu::log_determinant_view` | `MAG-D-003` | Verified |
| `lu::solve_complex` | `MAG-D-004` | Verified |
| `lu::solve_complex_view` | `MAG-D-004` | Verified |
| `lu::inverse_complex` | `MAG-D-005` | Verified |
| `lu::inverse_complex_view` | `MAG-D-005` | Verified |
| `lu::determinant_complex` | `MAG-D-006` | Verified |
| `lu::determinant_complex_view` | `MAG-D-006` | Verified |
| `lu::solve_mixed_f64` | `MAG-D-007` | Verified |
| `lu::solve_mixed_f64_view` | `MAG-D-007` | Verified |
| `lu::solve_mixed_complex` | `MAG-D-008` | Verified |
| `lu::solve_mixed_complex_view` | `MAG-D-008` | Verified |

## Cholesky

| Function | Route ID | Status |
|---|---|---|
| `cholesky::decompose` | `MAG-D-009` | Verified |
| `cholesky::decompose_view` | `MAG-D-009` | Verified |
| `cholesky::solve` | `MAG-D-010` | Verified |
| `cholesky::solve_view` | `MAG-D-010` | Verified |
| `cholesky::solve_into` | `MAG-D-010` | Verified |
| `cholesky::inverse` | `MAG-D-011` | Verified |
| `cholesky::inverse_view` | `MAG-D-011` | Verified |
| `cholesky::decompose_complex` | `MAG-D-012` | Verified |
| `cholesky::decompose_complex_view` | `MAG-D-012` | Verified |
| `cholesky::solve_complex` | `MAG-D-013` | Verified |
| `cholesky::solve_complex_view` | `MAG-D-013` | Verified |
| `cholesky::inverse_complex` | `MAG-D-014` | Verified |
| `cholesky::inverse_complex_view` | `MAG-D-014` | Verified |

## QR

| Function | Route ID | Status |
|---|---|---|
| `qr::decompose` | `MAG-D-015` | Verified |
| `qr::decompose_view` | `MAG-D-015` | Verified |
| `qr::decompose_reduced` | `MAG-D-015` | Verified |
| `qr::decompose_with_pivoting` | `MAG-D-015` | Verified |
| `qr::decompose_complex` | `MAG-D-016` | Verified |
| `qr::decompose_complex_view` | `MAG-D-016` | Verified |
| `qr::solve_least_squares` | `MAG-D-017` | Verified |
| `qr::solve_least_squares_view` | `MAG-D-017` | Verified |

## SVD

| Function | Route ID | Status |
|---|---|---|
| `svd::decompose` | `MAG-D-018` | Verified |
| `svd::decompose_view` | `MAG-D-018` | Verified |
| `svd::decompose_with_tolerance` | `MAG-D-018` | Verified |
| `svd::decompose_truncated` | `MAG-D-018` | Verified |
| `svd::pseudo_inverse` | `MAG-D-018` | Verified |
| `svd::pseudo_inverse_into` | `MAG-D-018` | Verified |
| `svd::null_space` | `MAG-D-018` | Verified |
| `svd::decompose_complex` | `MAG-D-019` | Verified |
| `svd::decompose_complex_view` | `MAG-D-019` | Verified |

## Eigen

| Function | Route ID | Status |
|---|---|---|
| `eigen::symmetric` | `MAG-D-020` | Verified |
| `eigen::symmetric_view` | `MAG-D-020` | Verified |
| `eigen::generalized` | `MAG-D-021` | Verified |
| `eigen::generalized_view` | `MAG-D-021` | Verified |
| `eigen::nonsymmetric_complex` | `MAG-D-022` | Verified |
| `eigen::nonsymmetric_complex_view` | `MAG-D-022` | Verified |

## Batched Decompositions

| Function | Route ID | Status |
|---|---|---|
| `batched::lu` | `MAG-D-023` | Verified |
| `batched::lu_view` | `MAG-D-023` | Verified |
| `batched::cholesky` | `MAG-D-024` | Verified |
| `batched::cholesky_view` | `MAG-D-024` | Verified |
| `batched::qr` | `MAG-D-025` | Verified |
| `batched::qr_view` | `MAG-D-025` | Verified |
| `batched::svd` | `MAG-D-026` | Verified |
| `batched::svd_view` | `MAG-D-026` | Verified |
| `batched::symmetric_eigen` | `MAG-D-027` | Verified |
| `batched::symmetric_eigen_view` | `MAG-D-027` | Verified |

## Sylvester / Lyapunov Mixed

| Function | Route ID | Status |
|---|---|---|
| `sylvester::solve_sylvester_mixed_f64` | `MAG-D-028` | Verified |
| `sylvester::solve_sylvester_mixed_f64_view` | `MAG-D-028` | Verified |
| `sylvester::solve_lyapunov_mixed_f64` | `MAG-D-028` | Verified |
| `sylvester::solve_lyapunov_mixed_f64_view` | `MAG-D-028` | Verified |
| `sylvester::solve_sylvester_mixed_complex` | `MAG-D-029` | Verified |
| `sylvester::solve_sylvester_mixed_complex_view` | `MAG-D-029` | Verified |
| `sylvester::solve_lyapunov_mixed_complex` | `MAG-D-029` | Verified |
| `sylvester::solve_lyapunov_mixed_complex_view` | `MAG-D-029` | Verified |

## Schur (Composed)

| Function | Route ID | Status |
|---|---|---|
| `schur::compute_schur` | `MAG-D-030` | Verified |
| `schur::compute_schur_view` | `MAG-D-030` | Verified |
| `schur::compute_schur_into` | `MAG-D-030` | Verified |
| `schur::compute_schur_into_view` | `MAG-D-030` | Verified |
| `schur::compute_schur_with_workspace_into` | `MAG-D-030` | Verified |
| `schur::compute_schur_complex` | `MAG-D-031` | Verified |
| `schur::compute_schur_complex_view` | `MAG-D-031` | Verified |
| `schur::compute_schur_complex_into` | `MAG-D-031` | Verified |
| `schur::compute_schur_complex_into_view` | `MAG-D-031` | Verified |
| `schur::compute_schur_complex_with_workspace_into` | `MAG-D-031` | Verified |

## Polar (Composed)

| Function | Route ID | Status |
|---|---|---|
| `polar::compute_polar` | `MAG-D-032` | Verified |
| `polar::compute_polar_view` | `MAG-D-032` | Verified |
| `polar::compute_polar_complex` | `MAG-D-033` | Verified |
| `polar::compute_polar_complex_view` | `MAG-D-033` | Verified |

## Matrix Functions (Composed)

| Function | Route ID | Status |
|---|---|---|
| `matrix_functions::matrix_exp_eigen` | `MAG-D-034` | Verified |
| `matrix_functions::matrix_exp_eigen_view` | `MAG-D-034` | Verified |
| `matrix_functions::matrix_exp_eigen_complex` | `MAG-D-035` | Verified |
| `matrix_functions::matrix_exp_eigen_complex_view` | `MAG-D-035` | Verified |
| `matrix_functions::matrix_log_eigen` | `MAG-D-036` | Verified |
| `matrix_functions::matrix_log_eigen_view` | `MAG-D-036` | Verified |
| `matrix_functions::matrix_log_eigen_into` | `MAG-D-036` | Verified |
| `matrix_functions::matrix_log_eigen_with_workspace_into` | `MAG-D-036` | Verified |
| `matrix_functions::matrix_log_eigen_complex` | `MAG-D-037` | Verified |
| `matrix_functions::matrix_log_eigen_complex_view` | `MAG-D-037` | Verified |
| `matrix_functions::matrix_log_eigen_complex_into` | `MAG-D-037` | Verified |
| `matrix_functions::matrix_log_eigen_complex_with_workspace_into` | `MAG-D-037` | Verified |
| `matrix_functions::matrix_log_svd` | `MAG-D-038` | Verified |
| `matrix_functions::matrix_log_svd_view` | `MAG-D-038` | Verified |
| `matrix_functions::matrix_log_svd_into` | `MAG-D-038` | Verified |
| `matrix_functions::matrix_log_svd_with_workspace_into` | `MAG-D-038` | Verified |
| `matrix_functions::matrix_log_svd_complex` | `MAG-D-039` | Verified |
| `matrix_functions::matrix_log_svd_complex_view` | `MAG-D-039` | Verified |
| `matrix_functions::matrix_log_svd_complex_into` | `MAG-D-039` | Verified |
| `matrix_functions::matrix_log_svd_complex_with_workspace_into` | `MAG-D-039` | Verified |
| `matrix_functions::matrix_power` | `MAG-D-040` | Verified |
| `matrix_functions::matrix_power_view` | `MAG-D-040` | Verified |
| `matrix_functions::matrix_power_into` | `MAG-D-040` | Verified |
| `matrix_functions::matrix_power_with_workspace_into` | `MAG-D-040` | Verified |
| `matrix_functions::matrix_power_complex` | `MAG-D-041` | Verified |
| `matrix_functions::matrix_power_complex_view` | `MAG-D-041` | Verified |
| `matrix_functions::matrix_power_complex_into` | `MAG-D-041` | Verified |
| `matrix_functions::matrix_power_complex_with_workspace_into` | `MAG-D-041` | Verified |
| `matrix_functions::matrix_sign` | `MAG-D-042` | Verified |
| `matrix_functions::matrix_sign_view` | `MAG-D-042` | Verified |
| `matrix_functions::matrix_sign_into` | `MAG-D-042` | Verified |
| `matrix_functions::matrix_sign_with_workspace_into` | `MAG-D-042` | Verified |
| `matrix_functions::matrix_sign_complex` | `MAG-D-043` | Verified |
| `matrix_functions::matrix_sign_complex_view` | `MAG-D-043` | Verified |
| `matrix_functions::matrix_sign_complex_into` | `MAG-D-043` | Verified |
| `matrix_functions::matrix_sign_complex_with_workspace_into` | `MAG-D-043` | Verified |

## Sparse MAGMA APIs

| Function | Route ID | Status |
|---|---|---|
| `sparse::matvec_magma_f64_view` | `MAG-S-001` | Verified |
| `sparse::matvec_magma_f64_view_into` | `MAG-S-001` | Verified |
| `sparse::matvec_magma_f32_view` | `MAG-S-002` | Verified |
| `sparse::matvec_magma_f32_view_into` | `MAG-S-002` | Verified |
| `sparse::matmat_dense_magma_f64_view` | `MAG-S-003` | Verified |
| `sparse::matmat_dense_magma_f64_view_into` | `MAG-S-003` | Verified |
| `sparse::matmat_dense_magma_f32_view` | `MAG-S-004` | Verified |
| `sparse::matmat_dense_magma_f32_view_into` | `MAG-S-004` | Verified |
| `sparse::conjugate_gradient_magma_f64_view` | `MAG-S-005` | Verified |
| `sparse::conjugate_gradient_magma_f32_view` | `MAG-S-006` | Verified |
| `sparse::pcg_jacobi_magma_f64_view` | `MAG-S-007` | Verified |
| `sparse::pcg_jacobi_magma_f32_view` | `MAG-S-008` | Verified |
| `sparse::gmres_magma_f64_view` | `MAG-S-009` | Verified |
| `sparse::gmres_magma_f32_view` | `MAG-S-010` | Verified |
| `sparse::gmres_ilu0_magma_f64_view` | `MAG-S-011` | Verified |
| `sparse::gmres_ilu0_magma_f32_view` | `MAG-S-012` | Verified |
| `sparse::bicgstab_magma_f64_view` | `MAG-S-013` | Verified |
| `sparse::bicgstab_magma_f32_view` | `MAG-S-014` | Verified |
| `sparse::bicgstab_ilu0_magma_f64_view` | `MAG-S-015` | Verified |
| `sparse::bicgstab_ilu0_magma_f32_view` | `MAG-S-016` | Verified |
