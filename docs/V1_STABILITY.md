# V1 Stability Contract

Last updated: 2026-03-02

## Purpose

This document defines what must be true to claim `nabled` is **100% stable for v1**.

This is a hard contract, not a roadmap.

## Ordered Gate

A v1-stable declaration requires all items below, in order:

1. Tensor API contract is explicit and fully implemented for the required surface.
2. GPU kernel contract is explicit and fully implemented for the required surface.
3. Mixed provider/backend/kernel behavior is deterministic and documented.
4. CI/local checks enforce required feature/build permutations.
5. No-surprises audit (allocation, fallback, errors, docs) passes.

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

1. `matmat_with_backend_f32::<CudaBackend>`
2. `matvec_with_backend_f32::<CudaBackend>`
3. `batched_matmat_with_backend_f32::<CudaBackend>`
4. `tensor_batched_matmul_last_two_with_backend::<CudaBackend, f32>`

Runtime note: on systems without a usable GPU device, these return `AcceleratorError::DeviceUnavailable`.

### CUDA `f64`/complex and tensor-op fallback behavior

For CUDA-dispatched kernels outside the bounded `f32` GPU-native surface:

1. The backend keeps capability stable and executes explicit CPU fallback paths.
2. Only true input/shape/runtime execution errors are returned.

This unified behavior is deliberate and tested.

## Mixed Execution Determinism Contract

1. `Provider` (internal vs `openblas-system`) is compile-time selected and applies to decomposition-style code paths.
2. `Backend` and `Kernel` are compile-time selected and apply to operation-family kernel paths.
3. Provider choice does not require runtime provider branching.
4. Public APIs remain provider/backend agnostic and ndarray-native.
5. CUDA out-of-scope kernels use explicit CPU fallback paths to preserve public API capability.

## Required Feature/Build Matrix

The following combinations are required checks for v1 stability:

1. `--no-default-features`
2. `--features openblas-system`
3. `--features accelerator-rayon`
4. `--features accelerator-wgpu`
5. `--features "openblas-system accelerator-rayon"`
6. `--features "openblas-system accelerator-wgpu"`
7. `--all-features`

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

Current status: satisfied for v1 required surface.
