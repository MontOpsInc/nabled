# Arrow Support Matrix

Last updated: 2026-03-09

## Purpose

Track direct Arrow-ingress coverage against `nabled`'s public API surface.

This matrix is intentionally about **capability parity**, not 1:1 wrapper parity.

Questions it answers:

1. Can a user start from Arrow data directly for this domain?
2. If yes, does that path delegate into the same canonical ndarray-native execution path?
3. Is the result Arrow-native, ndarray-native, or mixed?
4. What is the next highest-value gap to close?

Status meanings:

1. `Full`: current direct Arrow-ingress coverage is sufficient for the domain's primary public
   workflows.
2. `Partial`: some direct Arrow-ingress workflows exist, but meaningful public capabilities remain
   unwrapped.
3. `None`: no direct Arrow-ingress surface exists yet in `nabled::arrow`.

## Current Matrix

| Domain | Direct Arrow ingress | Current Arrow surface | Result contract | Execution parity | Primary gap |
|--------|----------------------|-----------------------|-----------------|------------------|-------------|
| `vector` | Partial | `dot`, `cosine_similarity`, `l2_norm`, `cosine_distance`, pairwise distance/cosine, `batched_dot` | Arrow scalar / Arrow vector / Arrow dense matrix | Yes; delegates to canonical vector view APIs | Complex vector workflows |
| `matrix` | Partial | `matvec`, `matmat`, `batched_row_matvec` | Arrow vector / Arrow dense matrix | Yes; delegates to canonical matrix view APIs | Complex dense kernels and additional dense helper families |
| `lu` | Partial | `decompose_*`, `solve_*`, `inverse_*`, `determinant_*`, `log_determinant_*` | ndarray-native decomposition result, Arrow vector solve result, Arrow dense inverse result | Yes; delegates to canonical LU view/public APIs | Complex LU ingress and broader factorization-reuse contracts |
| `cholesky` | Partial | `decompose_*`, `solve_*`, `inverse_*` | ndarray-native decomposition result, Arrow vector solve result, Arrow dense inverse result | Yes; delegates to canonical Cholesky view/public APIs | Complex Cholesky ingress and broader SPD convenience |
| `qr` | Partial | `decompose_*`, `solve_least_squares_*`, `reconstruct_*` | ndarray-native decomposition result, Arrow vector solve result, Arrow dense reconstruction result | Yes; delegates to canonical QR view/public APIs | Reduced/pivoted/complex QR ingress |
| `svd` | Partial | `decompose_*`, truncated/toleranced SVD, `pseudo_inverse_*`, `null_space_*` | ndarray-native SVD result, Arrow dense matrix results where natural | Yes; delegates to canonical SVD view/public APIs | Complex SVD ingress |
| `eigen` | Partial | symmetric/generalized/nonsymmetric/balancing/bi-eigen real wrappers | ndarray-native eigen result, mixed Arrow+ndarray where natural | Yes; delegates to canonical eigen view/public APIs | Complex Arrow eigen ingress |
| `schur` | Partial | `compute_*` | ndarray-native Schur result | Yes; delegates to canonical Schur view/public APIs | Complex Arrow Schur ingress |
| `matrix_functions` | Partial | `exp_*`, `exp_eigen_*`, `log_taylor_*`, `log_eigen_*`, `log_svd_*`, `power_*`, `sign_*` | Arrow dense matrix | Yes; delegates to canonical matrix-functions view/public APIs | Complex Arrow matrix-function ingress |
| `polar` | Partial | `compute_*` | ndarray-native polar result | Yes; delegates to canonical polar view/public APIs | Complex Arrow polar ingress |
| `sparse` | Partial | CSR columns/extension `matvec`, dense `matmat`, sparse LU factor/solve, Jacobi/Gauss-Seidel/CG/PCG solves | Arrow dense vector / Arrow dense matrix / ndarray-native factorization | Yes; delegates to canonical sparse view APIs | Remaining sparse solver breadth, reuse/preconditioner depth, complex sparse ingress |
| `triangular` | Partial | lower/upper vector solve and dense-matrix RHS solve (`f32`, `f64`) | Arrow vector / Arrow dense matrix | Yes; delegates to canonical triangular view/public APIs | Complex triangular ingress |
| `batched` | Partial | batched QR/SVD/LU/Cholesky/symmetric eigen | ndarray-native decomposition results | Yes; delegates to canonical batched APIs | Additional batch families and complex ingress |
| `tensor` | Partial | fixed-shape tensor reductions, normalization, batched dot, permutation, contraction, batched matmul, cube kernels | Arrow fixed-shape tensor | Yes; delegates to canonical tensor view APIs | Higher-rank/decomposition-family tensor workflows and complex ingress |
| `iterative` | Partial | dense `conjugate_gradient_*`, `gmres_*` | Arrow dense vector | Yes; delegates to canonical iterative public APIs | Complex iterative Arrow ingress |
| `jacobian` | Partial | `numerical_jacobian`, `numerical_jacobian_central`, `numerical_gradient`, `numerical_hessian` | Arrow vector / Arrow dense matrix | Yes; delegates to canonical Jacobian public APIs | Complex derivative ingress |
| `optimization` | Partial | line search, gradient descent, Adam, momentum, RMSProp, projected GD, SGD, BFGS | Arrow scalar / Arrow dense vector | Yes; delegates to canonical optimization public APIs | Complex optimization ingress |
| `pca` | Partial | `compute_*`, `transform_*`, `inverse_transform_*` | ndarray-native PCA result, Arrow dense matrix outputs where natural | Yes; delegates to canonical PCA public APIs | Complex PCA ingress |
| `regression` | Partial | `linear_regression_*` | ndarray-native regression result | Yes; delegates to canonical regression public APIs | Complex regression ingress |
| `stats` | Partial | means, centering, covariance, correlation (`f32`, `f64`) | Arrow vector / Arrow dense matrix | Yes; delegates to canonical stats public APIs | Complex stats ingress |

## Prioritization Rules

Expand the Arrow surface in this order:

1. Domains where direct Arrow ingress is natural and execution parity is easy to preserve.
2. Domains where Arrow users would otherwise be blocked from major `nabled` capabilities.
3. Domains where outputs can remain ndarray-native without surprise.
4. Domains where result mapping to Arrow is explicit and cheap.

Do **not** prioritize:

1. Low-value overload mirroring (`_view`, `_into`, workspace variants) when one Arrow ingress
   workflow can already reach the same algorithm.
2. Arrow wrappers that would need hidden materialization-heavy conversions.
3. Arrow-specific algorithm forks.

## Expansion Gate

Before adding a new Arrow wrapper, confirm:

1. It delegates to the canonical ndarray-native public/view entrypoint.
2. Provider/backend/kernel behavior is inherited rather than reimplemented.
3. The input Arrow shape contract is explicit.
4. The output contract is explicit.
5. Tests cover both plain `arrow` and provider-enabled `arrow` where relevant.
