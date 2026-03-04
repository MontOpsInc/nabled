# Decisions

## Locked Decisions

1. Canonical compute substrate is `ndarray`.
2. Public APIs are pure numerical APIs over ndarray types.
3. `nabled` does not depend on Arrow types.
4. Workspace structure is required for long-term scale.
5. No hidden data conversion in hot compute paths.
6. Quality gates remain strict: pedantic linting, CI parity, and coverage >= 90%.
7. Backend selection is compile-time only; no runtime backend dispatch.
8. Default execution path is internal ndarray-native implementations.
9. Backend-specific behavior must not leak into public API names.
10. No legacy/backward-compatibility shims for unreleased APIs.
11. Decomposition-style APIs use concise domain naming (for example, `svd::decompose`).
12. Performance-critical kernels expose explicit allocation-control APIs (`*_into`) and optional reusable workspace types.
13. View/convenience APIs must not hide heap allocations without explicit rustdoc disclosure.
14. Dense-kernel tolerance and iteration defaults are centralized in one shared policy (`nabled-linalg::internal::DenseKernelPolicy`).
15. Delivery strategy is domain-first vertical slices: each domain is finalized with API, tests, benchmarks, and docs before expanding horizontally.
16. P2 kickoff is incremental: establish rank-3 tensor/cube APIs and compile-time accelerator contracts first, then add concrete GPU and future multi-node kernels.
17. Execution terminology is explicit and stable:
   - `Provider`: decomposition implementation source (for example, internal vs OpenBLAS-backed).
   - `Backend`: primitive-kernel execution target (for example, CPU, GPU).
   - `Kernel`: operation-family contract implemented by backends (for example, matrix-matrix multiply).
18. Provider and backend are orthogonal axes and may both be used within one public algorithm flow.
19. Kernel implementations do not directly invoke provider selection; orchestration of provider-backed decomposition and backend-backed kernels lives in domain APIs.
20. Provider selection remains compile-time via feature gating (`#[cfg]`) in domain code; no runtime provider-dispatch API is required.
21. Kernel-family scope is explicitly cataloged (`docs/KERNEL_CATALOG.md`) and treated as finite/planned work, not ad hoc expansion.
22. V1 tensor and GPU capability scope is explicitly bounded and version-locked in `docs/V1_STABILITY.md`; out-of-scope behavior must be explicit/typed, not implicit fallback.

## API Purity Model

1. A function accepts ndarray inputs and returns ndarray outputs (or scalar/error outputs).
2. Any additional controls are explicit function arguments.
3. No assumptions about calling context (database, Arrow, SQL, transport layer).

## Data Types

1. Vector: `Array1<T>`, `ArrayView1<'a, T>`, `ArrayViewMut1<'a, T>`.
2. Matrix: `Array2<T>`, `ArrayView2<'a, T>`, `ArrayViewMut2<'a, T>`.
3. Higher-rank tensors (future): `ArrayD<T>` and fixed-rank aliases as needed.
4. Complex support: `num_complex::Complex32/Complex64` where algorithms are mathematically valid.

## Near-Term Non-Goals

1. Cross-library interop adapters.
2. Python bindings.
3. Arrow integration inside `nabled`.

These are deferred until the ndarray-first core is complete and stable.

## Provider and Backend Contract

1. `blas` is a baseline feature for enabling BLAS-accelerated ndarray paths where available.
2. LAPACK acceleration is provider-driven, not a separate runtime backend layer.
3. Provider scope includes four LAPACK provider features: `openblas-system`, `openblas-static`, `netlib-system`, and `netlib-static`.
4. Provider features imply `blas` so users do not have to compose low-level flags manually.
5. LAPACK-accelerated code should be gated by feature selection, not by hardcoded `target_os` branching.
6. Current platform intent is macOS and Linux first; Windows support is deferred.
7. Backend acceleration is compile-time feature-gated and operation-specific.
8. Backend acceleration does not imply provider acceleration, and provider acceleration does not imply backend acceleration.
