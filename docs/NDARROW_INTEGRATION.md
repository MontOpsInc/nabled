# Ndarrow Integration

Last updated: 2026-03-09

## Purpose

Define how `nabled` collaborates with `ndarrow` without compromising `nabled`'s ndarray-native core.

## Contract

1. `nabled-core`, `nabled-linalg`, and `nabled-ml` remain Arrow-free.
2. Arrow awareness lives only in facade crate `crates/nabled`.
3. Arrow support is explicit and opt-in via feature `arrow`.
4. Inbound Arrow -> ndarray bridging uses `ndarrow` zero-copy views wherever the underlying
   `nabled` API already accepts ndarray views.
5. Outbound ndarray -> Arrow conversion is provided only where the result shape maps naturally to
   Arrow (`PrimitiveArray`, `FixedSizeListArray`, canonical fixed-shape tensor extension).
6. Multi-output decomposition results remain in `nabled`'s ndarray-native result structs unless a
   dedicated Arrow result contract is explicitly introduced later.
7. Arrow is an ingress/storage format, not a separate compute engine.
8. Arrow adapters must delegate to the canonical ndarray-native public/view entrypoint for the
   operation so providers, backends, kernels, routing, and fallback policy remain shared with
   normal ndarray users.

## Admission Rules

Future Arrow wrappers are admitted only when all of the following are true:

1. The lower `nabled` API is already ndarray-view-native or can be widened generically without
   introducing Arrow-specific code in lower crates.
2. The Arrow input contract is natural and structurally explicit (`PrimitiveArray`, dense
   `FixedSizeListArray`, canonical CSR extension, canonical fixed-shape tensor extension, etc.).
3. The adapter does not introduce hidden allocation-heavy conversions in the hot path.
4. The adapter delegates to the same canonical ndarray-native public/view entrypoint that owns
   provider/backend/kernel behavior for ndarray callers.
5. The Arrow result contract is explicit:
   - Arrow-native output only when the shape maps naturally and cheaply to Arrow.
   - ndarray-native result structs are acceptable when they are the natural contract.

## Parity Model

Arrow expansion targets capability parity, not 1:1 signature parity.

This means:

1. Arrow users should be able to reach the same numerical algorithms and execution machinery as
   ndarray users for supported workflows.
2. Arrow wrappers do **not** need to mirror every `_view`, `_into`, workspace, internal, or
   provider/backend-specific overload.
3. A single well-chosen Arrow wrapper per workflow is preferred over duplicating the full lower
   API surface.
4. If a workflow cannot yet be expressed with a natural Arrow boundary and explicit contract, it
   should remain unwrapped until that contract is clear.

## Current Surface

The current facade-level Arrow adapter module is `nabled::arrow` behind feature `arrow`.

Implemented surface:

1. `nabled::arrow::vector`
   - `dot`
   - `cosine_similarity`
   - `l2_norm`
   - `cosine_distance`
   - `pairwise_l2_distance`
   - `pairwise_cosine_similarity`
   - `pairwise_cosine_distance`
   - `batched_dot`
2. `nabled::arrow::matrix`
   - `matvec`
   - `matmat`
   - `batched_row_matvec`
3. `nabled::arrow::lu`
   - `decompose_f32`
   - `decompose_f64`
   - `solve_f32`
   - `solve_f64`
   - `inverse_f32`
   - `inverse_f64`
   - `determinant_f32`
   - `determinant_f64`
   - `log_determinant_f32`
   - `log_determinant_f64`
4. `nabled::arrow::cholesky`
   - `decompose_f32`
   - `decompose_f64`
   - `solve_f32`
   - `solve_f64`
   - `inverse_f32`
   - `inverse_f64`
5. `nabled::arrow::qr`
   - `decompose_f32`
   - `decompose_f64`
   - `solve_least_squares_f32`
   - `solve_least_squares_f64`
   - `reconstruct_f32`
   - `reconstruct_f64`
6. `nabled::arrow::svd`
   - `decompose_f32`
   - `decompose_f64`
   - `decompose_truncated_f32`
   - `decompose_truncated_f64`
   - `decompose_with_tolerance_f32`
   - `decompose_with_tolerance_f64`
   - `pseudo_inverse_f32`
   - `pseudo_inverse_f64`
   - `null_space_f32`
   - `null_space_f64`
7. `nabled::arrow::eigen`
   - `symmetric_f32`
   - `symmetric_f64`
   - `generalized_f32`
   - `generalized_f64`
   - `nonsymmetric_f32`
   - `nonsymmetric_f64`
   - `balance_nonsymmetric_f32`
   - `balance_nonsymmetric_f64`
   - `nonsymmetric_bi_f32`
   - `nonsymmetric_bi_f64`
8. `nabled::arrow::schur`
   - `compute_f32`
   - `compute_f64`
9. `nabled::arrow::polar`
   - `compute_f32`
   - `compute_f64`
10. `nabled::arrow::matrix_functions`
   - `exp_*`
   - `exp_eigen_*`
   - `log_taylor_*`
   - `log_eigen_*`
   - `log_svd_*`
   - `power_*`
   - `sign_*`
11. `nabled::arrow::orthogonalization`
   - `gram_schmidt_*`
   - `gram_schmidt_classic_*`
12. `nabled::arrow::triangular`
   - lower/upper vector solve (`f32`, `f64`)
   - lower/upper dense-matrix RHS solve (`f32`, `f64`)
13. `nabled::arrow::sparse`
   - CSR columns/extension -> dense `matvec`
   - CSR columns/extension -> dense `matmat`
   - CSR columns/extension -> direct sparse LU factorization
   - CSR columns/extension -> direct sparse LU solve
   - CSR columns/extension -> Jacobi/Gauss-Seidel/CG/PCG solves
14. `nabled::arrow::iterative`
   - dense `conjugate_gradient_*`
   - dense `gmres_*`
15. `nabled::arrow::jacobian`
   - `numerical_jacobian`
   - `numerical_jacobian_central`
   - `numerical_gradient`
   - `numerical_hessian`
16. `nabled::arrow::optimization`
   - `backtracking_line_search`
   - `gradient_descent`
   - `adam`
   - `momentum_descent`
   - `rmsprop`
   - `projected_gradient_descent_box`
   - `stochastic_gradient_descent`
   - `bfgs`
17. `nabled::arrow::pca`
   - `compute_*`
   - `transform_*`
   - `inverse_transform_*`
18. `nabled::arrow::regression`
   - `linear_regression_*`
19. `nabled::arrow::stats`
   - `column_means_*`
   - `center_columns_*`
   - `covariance_matrix_*`
   - `correlation_matrix_*`
20. `nabled::arrow::tensor`
   - canonical fixed-shape tensor `sum_last_axis`
   - `l2_norm_last_axis`
   - `normalize_last_axis`
   - `batched_dot_last_axis`
   - `permute_axes`
   - `contract_axes`
   - `batched_matmul_last_two`
   - `cube_matvec`
   - `cube_matmat`
21. `nabled::arrow::batched`
   - batched `qr_*`
   - batched `svd_*`
   - batched `lu_*`
   - batched `cholesky_*`
   - batched `symmetric_eigen_*`

## Current Data Mapping Rules

1. Dense vectors: `PrimitiveArray<T>` -> `ArrayView1<T>`
2. Dense matrices: `FixedSizeListArray` -> `ArrayView2<T>`
3. Sparse CSR:
   - Arrow columns (`List<UInt32>` + `List<T>`) -> `ndarrow::CsrView<T>` -> `nabled::sparse::CsrMatrixView`
   - `ndarrow.csr_matrix` extension -> same route
4. Fixed-shape tensors:
   - canonical `arrow.fixed_shape_tensor` storage -> `ArrayViewD<T>`

## Scope Boundaries

This integration does not currently provide:

1. Complex Arrow facade adapters; under the current contract there is no native Arrow complex type
   or locked canonical complex extension contract for `nabled`.
2. Arrow-native decomposition result structs for multi-output workflows; many results remain
   ndarray-native when that is the natural contract.
3. Arrow integration inside lower crates.
4. Additional Arrow wrappers only where the boundary/result contract is still not yet explicit or
   worth shipping.

## Expansion Rules

Future Arrow surface should expand only when:

1. The underlying `nabled` API is already ndarray-view-native or can be widened generically without
   introducing Arrow-specific code in lower crates.
2. The Arrow result contract is explicit and unsurprising.
3. Docs, tests, and feature-matrix checks are updated in the same change set.
4. The new wrapper moves the Arrow support matrix toward capability parity rather than adding
   low-value overload mirroring.

## Tracking

1. `docs/ARROW_SUPPORT_MATRIX.md` is the direct-ingress coverage ledger for the current public API.
2. Add Arrow wrappers by closing the highest-value `None` or `Partial` rows in that matrix first.
