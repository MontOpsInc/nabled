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

## Test Coverage Enforcement

Capability smoke tests are enforced in [tests/integration.rs](../tests/integration.rs):

1. `test_backend_tier_a_baseline_capability_smoke`
2. `test_backend_tier_a_lapack_capability_smoke` (Linux + `lapack-kernels`)

