# Arrow Support Matrix

Last updated: 2026-03-11

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
| `vector` | Full | real + complex vector primitives, pairwise kernels, `batched_dot` | Arrow scalar / Arrow vector / Arrow dense matrix | Yes; delegates to canonical vector view APIs | No direct-ingress gap under the current explicit contract |
| `matrix` | Full | real + complex dense kernels, row-batched matvec, batched matmul/broadcast families | Arrow vector / Arrow dense matrix / Arrow fixed-shape tensor where natural | Yes; delegates to canonical matrix view APIs | No direct-ingress gap under the current explicit contract |
| `lu` | Full | `decompose_*`, `solve_*`, `inverse_*`, `determinant_*`, `log_determinant_*`, complex solve/inverse/determinant | ndarray-native decomposition result, Arrow vector solve result, Arrow dense inverse result | Yes; delegates to canonical LU view/public APIs | No direct-ingress gap under the current explicit contract |
| `cholesky` | Full | `decompose_*`, `solve_*`, `inverse_*`, complex decomposition/solve/inverse | ndarray-native decomposition result, Arrow vector solve result, Arrow dense inverse result | Yes; delegates to canonical Cholesky view/public APIs | No direct-ingress gap under the current explicit contract |
| `qr` | Full | full/reduced/pivoted QR, least-squares, reconstruction, condition number, complex QR | ndarray-native decomposition result, Arrow vector solve result, Arrow dense reconstruction result | Yes; delegates to canonical QR view/public APIs | No direct-ingress gap under the current explicit contract |
| `svd` | Full | full/truncated/toleranced SVD, pseudo-inverse, null-space, reconstruction, rank/condition number, complex SVD | ndarray-native SVD result, Arrow dense matrix results where natural | Yes; delegates to canonical SVD view/public APIs | No direct-ingress gap under the current explicit contract |
| `eigen` | Full | symmetric/generalized/nonsymmetric/balancing/bi-eigen real wrappers plus complex nonsymmetric ingress | ndarray-native eigen result, mixed Arrow+ndarray where natural | Yes; delegates to canonical eigen view/public APIs | No direct-ingress gap under the current explicit contract |
| `schur` | Full | real + complex Schur decomposition | ndarray-native Schur result | Yes; delegates to canonical Schur view/public APIs | No direct-ingress gap under the current explicit contract |
| `matrix_functions` | Full | `exp_*`, `exp_eigen_*`, `log_taylor_*`, `log_eigen_*`, `log_svd_*`, `power_*`, `sign_*`, complex variants where public APIs exist | Arrow dense matrix | Yes; delegates to canonical matrix-functions view/public APIs | No direct-ingress gap under the current explicit contract |
| `polar` | Full | real + complex polar decomposition | ndarray-native polar result | Yes; delegates to canonical polar view/public APIs | No direct-ingress gap under the current explicit contract |
| `sparse` | Full | CSR columns/extension `matvec`, dense/sparse `matmat`, transpose/CSR->CSC, batched matvec, factorization builders, preconditioner application, sparse LU factor/solve/reuse | Arrow dense vector / Arrow dense matrix / ndarray-native sparse matrices and factorizations | Yes; delegates to canonical sparse view APIs | No direct-ingress gap under the current explicit contract |
| `triangular` | Full | lower/upper vector solve and dense-matrix RHS solve (`f32`, `f64`, complex where public APIs exist) | Arrow vector / Arrow dense matrix | Yes; delegates to canonical triangular view/public APIs | No direct-ingress gap under the current explicit contract |
| `batched` | Full | batched QR/SVD/LU/Cholesky/symmetric eigen | ndarray-native decomposition results | Yes; delegates to canonical batched APIs | No direct-ingress gap under the current explicit contract |
| `tensor` | Full | fixed-shape tensor reductions/normalization/permutation/contraction/batched matmul, real+complex cube kernels, einsum, CP-ALS, HOSVD/HOOI/Tucker, TT-SVD and TT algebra/reconstruction | Arrow fixed-shape tensor, ndarray-native decomposition/network result structs where natural | Yes; delegates to canonical tensor view/public APIs | No direct-ingress gap under the current explicit contract |
| `iterative` | Full | dense `conjugate_gradient_*`, `gmres_*`, real + complex | Arrow dense vector | Yes; delegates to canonical iterative public APIs | No direct-ingress gap under the current explicit contract |
| `jacobian` | Full | `numerical_jacobian`, `numerical_jacobian_central`, `numerical_gradient`, `numerical_hessian` | Arrow vector / Arrow dense matrix | Yes; delegates to canonical Jacobian public APIs | No direct-ingress gap under the current explicit contract |
| `optimization` | Full | line search, gradient descent, Adam, momentum, RMSProp, projected GD, SGD, BFGS, real + complex | Arrow scalar / Arrow dense vector | Yes; delegates to canonical optimization public APIs | No direct-ingress gap under the current explicit contract |
| `pca` | Full | `compute_*`, `transform_*`, `inverse_transform_*`, real + complex | ndarray-native PCA result, Arrow dense matrix outputs where natural | Yes; delegates to canonical PCA public APIs | No direct-ingress gap under the current explicit contract |
| `regression` | Full | `linear_regression_*`, real + complex | ndarray-native regression result | Yes; delegates to canonical regression public APIs | No direct-ingress gap under the current explicit contract |
| `stats` | Full | means, centering, covariance, correlation, real + complex | Arrow vector / Arrow dense matrix | Yes; delegates to canonical stats public APIs | No direct-ingress gap under the current explicit contract |

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
