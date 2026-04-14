# Performance Contracts

Last updated: 2026-04-14

## Purpose

This document defines and records the allocation/copy contract for `nabled` APIs.

The goal is explicit: avoid hidden materialization in public convenience paths, and document any unavoidable internal allocation required by algorithm structure or provider APIs.

## Contract

1. View-first APIs (`*_view`) must not perform wrapper-level `to_owned()` materialization.
2. Hot paths should provide allocation-control forms (`*_into`, workspace variants) where practical.
3. Unavoidable allocations must be documented in this file (and in rustdoc where relevant).

## N-061 Copy-Elision Audit (Completed)

### Optimized avoidable materializations

1. `nabled-ml::iterative::gmres` now uses view-backed dot products for Krylov basis/Hessenberg algebra, removing intermediate owned buffers.
2. `nabled-ml::regression::linear_regression_impl` no longer performs an explicit `y.to_owned()` when computing residuals.
3. `nabled-linalg::internal::qr_gram_schmidt` now reuses a scratch vector instead of allocating per-column temporaries.
4. `nabled-linalg::qr::decompose_complex_internal` now reuses a scratch complex vector instead of allocating per-column temporaries.
5. `nabled-linalg::svd::decompose_internal` and `decompose_complex_internal` now avoid temporary owned right-singular column materializations where view math suffices.
6. Kernel-routing regression fixes restored no-hidden-allocation behavior for `*_into` paths in `vector`, `sparse`, `triangular`, and `tensor` domains.
7. Sparse GPU phase-1 composition (`sparse_matmat_dense_gpu_*`) now reuses a per-column output buffer and writes directly into the final dense output matrix, eliminating per-column temporary result allocations in wrapper code.
8. Sparse factorization-backed direct and iterative reuse paths now admit borrowed dense RHS `ArrayBase` views all the way through the Rust core, so the Python sparse bridge no longer has to clone dense RHS arrays before calling reusable `GMRES` / `BiCGSTAB` / sparse-LU factor workflows.
9. `pynabled` QR least-squares factor reuse now dispatches directly over borrowed `Q` / `R` / optional permutation views instead of rebuilding an owned Rust `QRResult` from Python arrays just to reach the existing solver path.
10. Tensor reconstruction/diagnostic helper paths now borrow result factor/core arrays directly through the Rust core: HOSVD/Tucker/CP/TT reconstruct helpers and CP diagnostics no longer rebuild owned Rust result structs from Python arrays, and CP diagnostics now compute residual metrics directly instead of allocating a full reconstructed tensor first.
11. Tensor-Train borrowed-core helper paths now materialize only the SVD work matrices they already need for TT orthogonalization/rounding sweeps, instead of first rebuilding an owned `TensorTrainResult` or requiring standard-layout TT core views at the Python/Arrow boundary.
12. Direct Python triangular solve `out=` paths now pass borrowed RHS views into generic mutable-output `nabled-linalg::triangular` helpers, so the binding no longer clones vector or matrix RHS/result arrays just to reuse caller-provided output buffers.
13. Owned tensor egress in `pynabled` no longer standardizes ndarray layout before NumPy handoff; owned Fortran/strided tensor results are now handed to NumPy with preserved strides instead of an extra full clone.
14. Direct Python stats and orthogonalization `out=` paths now write through shared Rust `*_into` helpers (`nabled-ml::stats` and `nabled-linalg::orthogonalization`) instead of forcing wrapper-level owned result allocation before handing arrays back to NumPy.
15. Factor-derived matrix-function `*_into` and workspace-backed direct-matrix paths now compose the current symmetric-eigen and SVD-backed outputs directly into caller-provided buffers through reusable scratch-backed matmul instead of allocating a full intermediate result and then copying it into `out`.
16. Python `polar_compute(..., out=...)` now decomposes once and writes `u` / `p` directly into caller-provided `PolarResult` buffers for both direct matrix inputs and typed `SvdResult` factor inputs instead of materializing an intermediate full polar result before copying.

### Unavoidable internal materializations

1. In-place decomposition kernels (for example LU/Schur/Polar and some Eigen paths) require one owned working matrix when input is provided as an immutable view.
2. Provider-backed calls through `ndarray-linalg` can require owned arrays due provider trait/method signatures (not wrapper-level conversion policy).
3. Shape-changing outputs (for example reduced/truncated decomposition outputs) allocate result arrays by API contract.
4. Current `wgpu` kernels stage host input buffers to device and read output buffers back to host memory per invocation; this host↔device transfer is expected for the current ndarray-owned public API contract.
5. Opt-in MAGMA sparse APIs (`matvec_magma_*`, `matmat_dense_magma_*`) stage CSR/vector/dense host buffers and allocate provider/device workspace per invocation; this is required by MAGMA sparse C API contracts and is explicit to these MAGMA-only entrypoints.
6. Opt-in MAGMA mixed LU APIs (`solve_mixed_f64*`, `solve_mixed_complex*`) allocate provider work buffers and stage matrix/RHS/solution host↔device transfers per invocation; behavior is explicit and confined to mixed-precision APIs.

## V1 No-Surprises Audit Status

Audit status: passed for v1 required surface.

1. Wrapper-level hidden allocations in `*_into` APIs have been removed from audited hot paths.
2. Remaining unavoidable allocations are algorithm/provider constrained and documented.
3. Feature-gated execution behavior (provider/backend/kernel) is now covered by explicit CI/local matrix checks.

## Enforcement

1. Keep `to_owned()` out of `*_view` wrappers unless explicitly documented and unavoidable.
2. Prefer view-native algebra (`dot` on views/slices) over temporary owned intermediates.
3. Validate all performance-contract changes under strict gates with `just checks`.
