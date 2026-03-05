# V1 Stability Contract

Last updated: 2026-03-05

## Purpose

This document defines what must be true to claim `nabled` is **100% stable for v1**.

This is a hard contract, not a roadmap.

## Ordered Gate

A v1-stable declaration requires all items below, in order:

1. Tensor API contract is explicit and fully implemented for the required surface.
2. GPU kernel contract is explicit and fully implemented for the required surface.
3. Mixed provider/backend/kernel behavior is deterministic and documented.
4. CI/local checks enforce required feature/build permutations.
5. Complex parity is complete for higher-level ML/statistical domains (`stats`, `regression`, `pca`, `optimization`) with explicit real/complex API contracts.
6. No-surprises audit (allocation, fallback, errors, docs) passes.

## Required Tensor Surface (V1)

The required tensor surface for v1 is:

1. Rank-3 cube primitives (real/complex):
   - `cube_matvec*`
   - `cube_matmat*`
2. Higher-rank `ArrayD` primitives (real/complex where applicable):
   - `sum_last_axis*`
   - `contract_axes*`
   - `batched_matmul_last_two*`
3. Each required operation family above must provide explicit API forms:
   - allocating (`op`)
   - view-first (`op_view`)
   - explicit output (`op_into` or `op_view_into`)

Current status: satisfied.

## Required GPU Surface (V1)

GPU scope is intentionally explicit and bounded for v1.

### Supported (feature `accelerator-wgpu`)

1. `matmat_with_backend_f32::<GpuBackend>`
2. `matvec_with_backend_f32::<GpuBackend>`
3. `batched_matmat_with_backend_f32::<GpuBackend>`
4. `batched_row_matvec_with_backend_f32::<GpuBackend>`
5. `dot_with_backend_f32::<GpuBackend>`
6. `pairwise_l2_with_backend_f32::<GpuBackend>`
7. `pairwise_cosine_with_backend_f32::<GpuBackend>`
8. `tensor_batched_matmul_last_two_with_backend::<GpuBackend, f32>`
9. `tensor_contract_axes_with_backend::<GpuBackend, f32>` (single-axis kernel path)
10. `tensor_sum_last_axis_with_backend::<GpuBackend, f32>`

Runtime note:
1. Backend-dispatched GPU paths above preserve capability via explicit CPU fallback when a usable GPU is unavailable.
2. Direct low-level GPU entrypoints in `accelerator::gpu` may still return `AcceleratorError::DeviceUnavailable`.

### GPU backend `f64` conditional support and fallback behavior

With `accelerator-wgpu`, `GpuBackend<f64>` now attempts native GPU execution for:

1. `matmat_with_backend::<GpuBackend, f64>`
2. `matvec_with_backend::<GpuBackend, f64>`
3. `batched_matmat_with_backend::<GpuBackend, f64>`
4. `batched_row_matvec_with_backend::<GpuBackend, f64>`
5. `dot_with_backend::<GpuBackend, f64>`
6. `pairwise_l2_with_backend::<GpuBackend, f64>`
7. `pairwise_cosine_with_backend::<GpuBackend, f64>`
8. `tensor_batched_matmul_last_two_with_backend::<GpuBackend, f64>`
9. `tensor_contract_axes_with_backend::<GpuBackend, f64>`
10. `tensor_sum_last_axis_with_backend::<GpuBackend, f64>`

Runtime note:
1. Native `f64` GPU execution depends on device/driver support for `wgpu::Features::SHADER_F64`.
2. When unavailable, backend-dispatched calls keep capability stable via explicit CPU fallback.
3. Direct low-level `accelerator::gpu::*_f64` entrypoints may return `AcceleratorError::UnsupportedBackend(BackendKind::Gpu)` when `SHADER_F64` is unavailable.

Complex GPU-backend kernels remain out-of-scope for native GPU execution and use explicit CPU fallback paths.

## Mixed Execution Determinism Contract

1. `Provider` (internal vs selected LAPACK provider feature) is compile-time selected and applies to decomposition-style code paths.
2. `Backend` and `Kernel` are compile-time selected and apply to operation-family kernel paths.
3. Provider choice does not require runtime provider branching.
4. Public APIs remain provider/backend agnostic and ndarray-native.
5. GPU-backend out-of-scope kernels use explicit CPU fallback paths to preserve public API capability.

## Required Feature/Build Matrix

The following combinations are required checks for v1 stability:

1. `--no-default-features`
2. `--features openblas-system`
3. `--features netlib-system`
4. `--features accelerator-rayon`
5. `--features accelerator-wgpu`
6. `--features "openblas-system accelerator-rayon"`
7. `--features "openblas-system accelerator-wgpu"`

Static provider notes:
1. `openblas-static` and `netlib-static` are supported feature paths.
2. They require native build toolchains (for example `gcc`, `gfortran`, `make`) and are validated in targeted environments.

Enforcement:

1. Local: `just checks` (includes accelerator compile+test permutations).
2. CI: `check` + `test-unit` jobs enforce provider and accelerator matrix coverage.

## No-Surprises Audit Checklist

All must be true:

1. `*_into` APIs do not hide extra allocation work as a wrapper behavior.
2. View-first APIs document any unavoidable owned materialization when required by algorithm/provider constraints.
3. Provider-disabled/provider-enabled paths return expected domain errors for identical bad input classes.
4. GPU out-of-scope areas execute explicit CPU fallback kernels and preserve capability.
5. Capability docs (`CAPABILITY_MATRIX`, `KERNEL_CATALOG`, `EXECUTION_TRACKER`) agree with code behavior.

Current status: v1 blocker-complete for declared capability scope; proceed with `K-*` normalization and benchmark/performance hardening passes.
