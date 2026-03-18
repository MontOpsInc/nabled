---
name: pynabled Full Test Suite
overview: Add Python tests for all pynabled bindings across 14 new test files plus 2 existing file updates, covering QR, LU, Cholesky, eigen, polar, sylvester, triangular, matrix_functions, vector, batched, ml, iterative, sparse, and tensor domains.
todos: []
isProject: false
---

# pynabled Full Test Suite

## Scope

Add tests for every exposed pynabled function. Existing tests: [python/tests/test_svd.py](python/tests/test_svd.py), [python/tests/test_matrix.py](python/tests/test_matrix.py). All new tests follow the same pattern: `np.float64` C-contiguous arrays, `np.testing.assert_allclose` for numerical checks, `pytest` for structure.

---

## Test File Structure

```
python/tests/
  test_svd.py          # extend (3 new tests)
  test_matrix.py       # extend (1 new test)
  test_qr.py           # new
  test_lu.py           # new
  test_cholesky.py     # new
  test_eigen.py        # new
  test_polar.py        # new
  test_sylvester.py    # new
  test_triangular.py   # new
  test_matrix_functions.py  # new
  test_vector.py       # new
  test_batched.py      # new
  test_ml.py           # new
  test_iterative.py    # new
  test_sparse.py       # new
  test_tensor.py       # new
```

---

## 1. Extend Existing Files

### test_svd.py (add 3 tests)

- `test_svd_reconstruct_matrix`: decompose, reconstruct via `svd_reconstruct_matrix`, assert `recon ≈ a`
- `test_svd_condition_number`: use `svd_condition_number(u, s, vt)` on known matrix, assert finite positive
- `test_svd_null_space`: rank-deficient matrix (e.g. `[[1,2],[2,4]]`), assert null space shape and `A @ null_vec ≈ 0`

### test_matrix.py (add 1 test)

- `test_gram_schmidt_classic`: same as `gram_schmidt` but `gram_schmidt_classic`, columns orthonormal

---

## 2. New Test Files

### test_qr.py

- `test_qr_decompose`: `q, r, rank = qr_decompose(a)`, assert `q @ r ≈ a`, `q.T @ q ≈ I`
- `test_qr_solve_least_squares`: overdetermined `A` (5x3), `b` (5,), solve, assert `np.linalg.norm(A @ x - b)` small; optionally compare with `np.linalg.lstsq`

### test_lu.py

- `test_lu_decompose`: `l, u = lu_decompose(a)`, assert `l @ u ≈ a` (or `P @ l @ u` if pivoting; check binding returns L,U only)
- `test_lu_solve`: `a @ x = b`, assert `a @ lu_solve(a, b) ≈ b`
- `test_lu_inverse`: `a @ lu_inverse(a) ≈ I`
- `test_lu_determinant`: compare with `np.linalg.det(a)`

### test_cholesky.py

- `test_cholesky_decompose`: SPD matrix `a = np.eye(3) + 0.1 * np.ones((3,3))`, `l = cholesky_decompose(a)`, assert `l @ l.T ≈ a`
- `test_cholesky_solve`: `a @ x = b`, assert `a @ cholesky_solve(a, b) ≈ b`
- `test_cholesky_inverse`: `a @ cholesky_inverse(a) ≈ I`

### test_eigen.py

- `test_eigen_generalized`: `a`, `b` SPD, `vals, vecs = eigen_generalized(a, b)`, assert `a @ vecs ≈ b @ vecs @ diag(vals)`
- `test_eigen_nonsymmetric`: `vals_re, vals_im, schur_re, schur_im = eigen_nonsymmetric(a)`, assert shapes and Schur relation (complex reconstruction if needed)

### test_polar.py

- `test_polar_compute`: `u, p = polar_compute(a)`, assert `u @ p ≈ a`, `u.T @ u ≈ I`, `p` symmetric

### test_sylvester.py

- `test_sylvester_solve`: small A, B, C; `x = sylvester_solve(a, b, c)`, assert `a @ x + x @ b ≈ c`
- `test_lyapunov_solve`: `x = lyapunov_solve(a, q)`, assert `a @ x + x @ a.T ≈ q`

### test_triangular.py

- `test_triangular_solve_lower`: lower L, b; `x = triangular_solve_lower(l, b)`, assert `l @ x ≈ b`
- `test_triangular_solve_upper`: upper U, b; `x = triangular_solve_upper(u, b)`, assert `u @ x ≈ b`
- `test_triangular_solve_lower_matrix`: L, B matrix RHS
- `test_triangular_solve_upper_matrix`: U, B matrix RHS

### test_matrix_functions.py

- `test_matrix_exp`: `exp_a = matrix_exp(a)`, `log_exp = matrix_log_taylor(exp_a)`, assert `log_exp ≈ a` (or use `matrix_exp_eigen`)
- `test_matrix_exp_eigen`: `exp_a = matrix_exp_eigen(a)`, compare with `matrix_exp` on small matrix
- `test_matrix_log_taylor`: `log_a = matrix_log_taylor(a)` for suitable `a` (e.g. SPD), `exp(log_a) ≈ a`
- `test_matrix_log_eigen`, `test_matrix_log_svd`: similar roundtrip where applicable
- `test_matrix_power`: `a2 = matrix_power(a, 2)`, assert `a2 ≈ a @ a`
- `test_matrix_sign`: diagonalizable matrix, assert `sign(a) @ sign(a) ≈ I` or known structure

### test_vector.py

- `test_l2_norm`: `l2_norm([3,4]) == 5`
- `test_cosine_similarity`: unit vectors, assert `cosine_similarity(a,b)` in [-1,1]
- `test_pairwise_l2_distance`: two matrices (n,d), assert output shape (n, m)
- `test_pairwise_cosine_similarity`: same, assert output in [-1,1]

### test_batched.py

- `test_batched_row_matvec`: cube (B,m,n), vectors (B,n), output (B,m)
- `test_batched_matmat`: two cubes (B,m,k), (B,k,n), output (B,m,n)
- `test_batched_qr`: cube (2,3,3), list of (Q,R), each Q@R ≈ slice
- `test_batched_svd`: cube, list of (U,s,Vt), reconstruct each
- `test_batched_lu`: cube, list of (L,U)
- `test_batched_cholesky`: cube of SPD matrices
- `test_batched_symmetric_eigen`: cube of symmetric matrices

### test_ml.py

- `test_linear_regression`: `x` (n,1), `y = 2*x + 1`, assert coefficients ≈ [1, 2], r_squared high
- `test_compute_pca`: `x` (n,d), `components, ev, evr, mean, scores = compute_pca(x, n_components=2)`, assert shapes, `evr.sum() <= 1`
- `test_column_means`: assert `column_means(x).shape == (d,)`
- `test_center_columns`: centered matrix has zero column means
- `test_covariance_matrix`: assert symmetric, shape (d,d)
- `test_correlation_matrix`: assert symmetric, diag ≈ 1, values in [-1,1]

### test_iterative.py

- `test_conjugate_gradient`: SPD `a`, `b`, `x = conjugate_gradient(a, b)`, assert `a @ x ≈ b`
- `test_gmres`: general `a`, `b`, `x = gmres(a, b)`, assert `a @ x ≈ b`

### test_sparse.py

- `test_sparse_matvec`: build CSR from dense matrix (e.g. `scipy.sparse.csr_matrix` or manual indptr/indices/data), `sparse_matvec(nrows, ncols, indptr, indices, data, v)`, compare with dense `A @ v`
- `test_sparse_jacobi_solve`: small sparse SPD system, assert residual small
- `test_sparse_pcg_solve`: same, PCG for SPD

**CSR format**: `indptr` length `nrows+1`, `indices` column indices, `data` values. Use `np.array(..., dtype=np.int64)` for indptr/indices.

### test_tensor.py

- `test_tensor_cube_matvec`: cube (B,m,n), vectors (B,n), output (B,m)
- `test_tensor_cube_matmat`: two cubes (B,m,k), (B,k,n)
- `test_tensor_sum_last_axis`: reduce last axis
- `test_tensor_l2_norm_last_axis`: norm over last axis
- `test_tensor_normalize_last_axis`: normalized rows
- `test_tensor_batched_dot_last_axis`: batched dot
- `test_tensor_permute_axes`: 3D tensor, permute [0,1,2] -> [2,1,0], assert shape
- `test_tensor_contract_axes`: e.g. contract last axis of (2,3,4) with first of (4,5)
- `test_tensor_batched_matmul_last_two`: (B,m,k) @ (B,k,n)
- `test_tensor_hosvd3`: cube, `core, u0, u1, u2 = tensor_hosvd3(cube, r0, r1, r2)`
- `test_tensor_hosvd3_reconstruct`: `recon = tensor_hosvd3_reconstruct(core, u0, u1, u2)`, assert `recon ≈ cube` (with rank truncation)

---

## 3. Implementation Notes

- **Numerical tolerance**: Use `rtol=1e-10`, `atol=1e-14` where needed (e.g. orthonormality).
- **Random seeds**: `np.random.seed(42)` or `np.random.default_rng(42)` for reproducibility.
- **SPD matrices**: `a = np.eye(n) + 0.1 * np.outer(v, v)` or `a = x.T @ x + eps * I`.
- **Sparse CSR**: For `test_sparse_`*, construct minimal CSR (e.g. 3x3 diagonal) manually or via scipy.
- **LU binding**: [crates/pynabled/src/linalg/lu.rs](crates/pynabled/src/linalg/lu.rs) returns `(L, U)` from `decompose`; verify if pivoting is applied (P stored separately or folded into L).
- **Optional args**: `tolerance=None`, `max_iterations=None` for iterative/sparse; `n_components=None` for PCA.

---

## 4. Verification

After implementation:

```bash
maturin develop
pytest python/tests/ -v
```

All tests should pass. No changes to Rust bindings or `__init__.py` required.