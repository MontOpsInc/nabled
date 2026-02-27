# Backend Capability Matrix

This matrix tracks Tier A backend-kernel support status and is used to prevent accidental regressions.

## Tier A Domains

| Domain | Baseline Kernels (`nalgebra` + `ndarray`) | LAPACK Kernels (`lapack-kernels`, Linux) | Notes |
|---|---|---|---|
| SVD | ✅ | ✅ | `src/backend/svd.rs` |
| QR | ✅ | ✅ | `src/backend/qr.rs` |
| LU | ✅ | ✅ | `src/backend/lu.rs` |
| Cholesky | ✅ | ✅ | `src/backend/cholesky.rs` |
| Eigen | ✅ | ✅ | `src/backend/eigen.rs` |
| Schur | ✅ | ✅ | `src/backend/schur.rs` |
| Triangular Solve | ✅ | ➖ | `src/backend/triangular.rs` (baseline kernels used under lapack-enabled builds) |

Legend:

- `✅`: Implemented and covered by smoke tests.
- `➖`: Not a distinct LAPACK kernel path yet.

## Tier B Pilot

Tier B backend-kernel pilot status:

| Domain | Baseline Kernels (`nalgebra` + `ndarray`) | LAPACK Kernels (`lapack-kernels`, Linux) | Notes |
|---|---|---|---|
| Polar | ✅ | ✅ | `src/backend/polar.rs` |
| PCA | ✅ | ✅ | `src/backend/pca.rs` |
| Regression | ✅ | ✅ | `src/backend/regression.rs` |
| Sylvester/Lyapunov | ✅ | ✅ | `src/backend/sylvester.rs` |

## Test Coverage Enforcement

Tier A capability tests are enforced in [tests/backend_capability_matrix.rs](../tests/backend_capability_matrix.rs):

1. Baseline tests: `test_tier_a_*_baseline`
2. LAPACK tests: `test_tier_a_*_lapack` (Linux + `lapack-kernels`)
