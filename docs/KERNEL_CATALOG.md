# Kernel Catalog

Last updated: 2026-03-02

## Purpose

This document defines the complete kernel contract set for `nabled` v1 stabilization.

It answers two questions:
1. What kernel families exist in the architecture.
2. Which kernel families are currently only contracts vs fully wired through public APIs.

## Status Legend

- `Wired`: contract exists, backend dispatch exists, and public APIs use it.
- `Dispatch`: contract exists and backend dispatch exists, but public APIs are not yet routed.
- `Contract`: trait exists, but backend dispatch/route work is still pending.

## Dense Kernels

| Kernel ID | Trait | Status | Current Public API Wiring |
|---|---|---|---|
| `K-DENSE-001` | `MatMatKernel<T>` | Wired | `matrix::matmat` |
| `K-DENSE-002` | `MatVecKernel<T>` | Wired | `matrix::matvec` |
| `K-DENSE-003` | `BatchedMatMatKernel<T>` | Wired | `matrix::batched_matmat` |
| `K-DENSE-004` | `BatchedRowMatVecKernel<T>` | Wired | `matrix::batched_row_matvec` |
| `K-DENSE-005` | `TriangularSolveVecKernel<T>` | Wired | `triangular::solve_lower`, `triangular::solve_upper` |
| `K-DENSE-006` | `TriangularSolveMatKernel<T>` | Wired | `triangular::solve_lower_matrix`, `triangular::solve_upper_matrix` |

## Sparse Kernels

| Kernel ID | Trait | Status | Current Public API Wiring |
|---|---|---|---|
| `K-SPARSE-001` | `SparseMatVecKernel` | Wired | `sparse::matvec` |
| `K-SPARSE-002` | `SparseMatMatDenseKernel` | Wired | `sparse::matmat_dense` |
| `K-SPARSE-003` | `SparseMatMatSparseKernel` | Wired | `sparse::matmat_sparse` |

## Vector Kernels

| Kernel ID | Trait | Status | Current Public API Wiring |
|---|---|---|---|
| `K-VECTOR-001` | `DotKernel<T>` | Wired | `vector::dot` |
| `K-VECTOR-002` | `PairwiseL2Kernel` | Wired | `vector::pairwise_l2_distance` |
| `K-VECTOR-003` | `PairwiseCosineKernel` | Wired | `vector::pairwise_cosine_similarity` |

## Tensor Kernels

| Kernel ID | Trait | Status | Current Public API Wiring |
|---|---|---|---|
| `K-TENSOR-001` | `TensorContractKernel<T>` | Wired | `tensor::contract_axes` (`len==1` kernel path, multi-axis fallback) |
| `K-TENSOR-002` | `TensorBatchedMatMulKernel<T>` | Wired | `tensor::batched_matmul_last_two` |
| `K-TENSOR-003` | `TensorLastAxisReductionKernel<T>` | Wired | `tensor::sum_last_axis` |

## Done-State Definition For Kernel Model

Kernel-model completion for v1 requires all catalog entries above to be `Wired`.

## Provider/Backend Orchestration Rules (K-008 Baseline)

1. Provider selection remains compile-time (`internal` vs `openblas-system`) and applies to decomposition-style paths.
2. Backend/kernel selection remains compile-time and applies to operation-family kernels.
3. Domain APIs may use both axes in one flow, but kernels must not trigger provider selection directly.
4. Public APIs stay backend/provider agnostic; no provider/backend naming leaks into stable API names.
5. All execution uses ndarray-native data at API boundaries.
