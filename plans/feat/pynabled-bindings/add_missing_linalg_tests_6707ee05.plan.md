---
name: Add missing linalg tests
overview: Add unit tests for all uncovered pure linear algebra functions across LU, Cholesky, QR, SVD, matrix functions, and utils modules.
todos: []
isProject: false
---

# Add Tests for Uncovered Pure Linear Algebra Functions

## Summary

Add ~25 new tests across 6 source files to cover the previously untested pure linear algebra APIs. Tests follow existing patterns (e.g., `assert_relative_eq!` from `approx`, same structure as adjacent tests).

---

## 1. LU ([src/lu.rs](src/lu.rs))

**Add:** `test_ndarray_lu_inverse`

Mirror the existing `test_nalgebra_lu_inverse`: use a 2x2 invertible matrix, compute inverse via `ndarray_lu::inverse`, verify `A * A⁻¹` approximates identity (diagonal 1.0, off-diagonal 0.0). Reuse the same matrix as `test_ndarray_lu_solve`: `[[1,2],[3,4]]`.

---

## 2. Cholesky ([src/cholesky.rs](src/cholesky.rs))

**Add:** `test_ndarray_cholesky_inverse`

Mirror `test_nalgebra_cholesky_inverse`: build SPD matrix from L (e.g. `[[2,0],[1,3]]`), compute inverse via `ndarray_cholesky::inverse`, verify `A * A⁻¹` approximates identity.

---

## 3. QR ([src/qr.rs](src/qr.rs))

**Add 4 tests:**

- `**test_nalgebra_qr_with_pivoting`**: Call `nalgebra_qr::compute_qr_with_pivoting` on a 3x3 matrix. Assert Q orthogonal (Q^T Q = I), R upper triangular, and reconstruction matches original. Use same matrix as `test_nalgebra_qr_basic`.
- `**test_ndarray_qr_with_pivoting`**: Same logic using `ndarray_qr::compute_qr_with_pivoting` with ndarray input; convert/reconstruct as in `test_ndarray_qr_basic` (via `ndarray_to_nalgebra` and `nalgebra_qr::reconstruct_matrix`).
- `**test_ndarray_reduced_qr`**: Call `ndarray_qr::compute_reduced_qr` on 4x3 matrix; verify dimensions (Q 4x3, R 3x3) and Q^T Q = I.
- `**test_ndarray_least_squares**`: Call `ndarray_qr::solve_least_squares` with overdetermined system (e.g. A 4x2, b length 4). Verify solution matches nalgebra result or known answer (e.g. [1,1] for the doc example).

---

## 4. SVD ([src/svd.rs](src/svd.rs))

**Add:** `test_nalgebra_svd_with_tolerance`

`compute_svd_with_tolerance` delegates to `compute_svd` (see [src/svd.rs:86-97](src/svd.rs)), so a simple correctness test suffices: use a 2x2 matrix, call `compute_svd_with_tolerance(&matrix, 1e-10)`, verify reconstruction matches original (same as `test_nalgebra_svd_basic`). Optionally assert empty matrix returns `Err(SVDError::EmptyMatrix)`.

---

## 5. Matrix Functions ([src/matrix_functions.rs](src/matrix_functions.rs))

**Add 8 tests:**

**Nalgebra:**

- `**test_nalgebra_matrix_exp_taylor`**: Use small matrix (e.g. 2x2 with small entries so Taylor converges). Call `matrix_exp(&matrix, 50, 1e-10)`. Verify `exp(A)` approximately matches `matrix_exp_eigen` or a known result (e.g. identity gives e*I).
- `**test_nalgebra_matrix_log_taylor*`*: Matrix close to identity (e.g. I + 0.1*[[0,1],[1,0]]) so ||A - I|| < 1. Call `matrix_log_taylor(&matrix, 100, 1e-10)`. Verify `exp(log(A))` approximates A.
- `**test_nalgebra_matrix_log_svd*`*: Use SPD matrix (e.g. [[2,1],[1,2]]). Call `matrix_log_svd`, verify `exp(log_svd(A))` approximates A (or compare to `matrix_log_eigen`).

**Ndarray:**

- `**test_ndarray_matrix_log_eigen`**: SPD `Array2`. Call `ndarray_matrix_functions::matrix_log_eigen`, verify exp(log(A)) round-trip approximates original.
- `**test_ndarray_matrix_power`**: Diagonal matrix `[[1,0],[0,4]]`. Call `matrix_power(&matrix, 0.5)`, verify result^2 approximates original.
- `**test_ndarray_matrix_exp_taylor`**: Same pattern as nalgebra; use `ndarray_matrix_functions::matrix_exp`.
- `**test_ndarray_matrix_log_taylor**`: Same as nalgebra with Array2 input.
- `**test_ndarray_matrix_log_svd**`: Same as nalgebra with Array2 input.

---

## 6. Utils ([src/utils.rs](src/utils.rs))

**Add 3 tests:**

- `**test_spectral_norm`**: Identity 2x2 → spectral norm 1.0. Diagonal [[1,0],[0,3]] → spectral norm 3.0. Use `approx::assert_relative_eq!`.
- `**test_matrix_approx_eq`**: Two equal matrices → true. Two matrices differing by epsilon → false with small epsilon, true with larger epsilon. Shape mismatch → false.
- `**test_random_matrix`**: Call `random_matrix::<f64>(3, 4)`, verify shape is (3, 4). (Current impl returns zeros; test is for API/contract.)

---

## File-by-File Checklist


| File                                               | New tests |
| -------------------------------------------------- | --------- |
| [src/lu.rs](src/lu.rs)                             | 1         |
| [src/cholesky.rs](src/cholesky.rs)                 | 1         |
| [src/qr.rs](src/qr.rs)                             | 4         |
| [src/svd.rs](src/svd.rs)                           | 1         |
| [src/matrix_functions.rs](src/matrix_functions.rs) | 8         |
| [src/utils.rs](src/utils.rs)                       | 3         |
| **Total**                                          | **18**    |


---

## Notes

- All new tests go in existing `#[cfg(test)] mod tests` blocks.
- Use `approx::assert_relative_eq!` with `epsilon = 1e-10` (or 1e-8 for matrix exp/log round-trips).
- `ndarray_to_nalgebra` in [src/utils.rs](src/utils.rs) returns `DMatrix`; `ndarray_to_nalgebra` in qr.rs may be a different helper that returns `Result` – use the one from the qr module for QR tests.
- For `matrix_log_taylor`, matrix must satisfy `||A - I|| < 1`; use I + small perturbation (e.g. 0.01 or 0.1 scale).

