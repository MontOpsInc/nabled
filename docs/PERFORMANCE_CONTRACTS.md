# Performance Contracts

Last updated: 2026-03-01

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

### Unavoidable internal materializations

1. In-place decomposition kernels (for example LU/Schur/Polar and some Eigen paths) require one owned working matrix when input is provided as an immutable view.
2. Provider-backed calls through `ndarray-linalg` can require owned arrays due provider trait/method signatures (not wrapper-level conversion policy).
3. Shape-changing outputs (for example reduced/truncated decomposition outputs) allocate result arrays by API contract.

## Enforcement

1. Keep `to_owned()` out of `*_view` wrappers unless explicitly documented and unavoidable.
2. Prefer view-native algebra (`dot` on views/slices) over temporary owned intermediates.
3. Validate all performance-contract changes under strict gates with `just checks`.
