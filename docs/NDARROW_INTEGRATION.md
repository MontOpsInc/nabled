# Ndarrow Integration

Last updated: 2026-03-12

## Purpose

Define how `nabled` collaborates with `ndarrow` without compromising `nabled`'s ndarray-native core.

## Contract

1. `nabled-core`, `nabled-linalg`, and `nabled-ml` remain Arrow-free.
2. Arrow awareness lives only in facade crate `crates/nabled`.
3. Arrow support is explicit and opt-in via feature `arrow`.
4. Inbound Arrow -> ndarray bridging uses `ndarrow` zero-copy views wherever the underlying
   `nabled` API already accepts ndarray views.
5. Outbound ndarray -> Arrow conversion is provided only where the result shape maps naturally to
   Arrow (`PrimitiveArray`, `FixedSizeListArray`, canonical fixed-shape tensor extension).
6. Multi-output decomposition results remain in `nabled`'s ndarray-native result structs unless a
   dedicated Arrow result contract is explicitly introduced later.
7. Arrow is an ingress/storage format, not a separate compute engine.
8. Arrow adapters must delegate to the canonical ndarray-native public/view entrypoint for the
   operation so providers, backends, kernels, routing, and fallback policy remain shared with
   normal ndarray users.
9. Arrow interop is concept-first: each mathematical object family should have one canonical
   standalone ingress and one canonical `rows-of-X` batch carrier.
10. Standalone batching should prefer the same canonical `rows-of-X` carriers that `ndatafusion`
    will use later; ad hoc collections of standalone Arrow objects are not canonical batch forms.

## Admission Rules

Future Arrow wrappers are admitted only when all of the following are true:

1. The lower `nabled` API is already ndarray-view-native or can be widened generically without
   introducing Arrow-specific code in lower crates.
2. The Arrow input contract is natural and structurally explicit (`PrimitiveArray`, dense
   `FixedSizeListArray`, canonical CSR extension, canonical fixed-shape tensor extension, etc.).
3. The adapter does not introduce hidden allocation-heavy conversions in the hot path.
4. The adapter delegates to the same canonical ndarray-native public/view entrypoint that owns
   provider/backend/kernel behavior for ndarray callers.
5. The Arrow result contract is explicit:
   - Arrow-native output only when the shape maps naturally and cheaply to Arrow.
   - ndarray-native result structs are acceptable when they are the natural contract.

## Parity Model

Arrow expansion targets capability parity, not 1:1 signature parity.

This means:

1. Arrow users should be able to reach the same numerical algorithms and execution machinery as
   ndarray users for supported workflows.
2. Arrow wrappers do **not** need to mirror every `_view`, `_into`, workspace, internal, or
   provider/backend-specific overload.
3. A single well-chosen Arrow wrapper per workflow is preferred over duplicating the full lower
   API surface.
4. If a workflow cannot yet be expressed with a natural Arrow boundary and explicit contract, it
   should remain unwrapped until that contract is clear.

## Current Surface

The current facade-level Arrow adapter module is `nabled::arrow` behind feature `arrow`.

Direct Arrow ingress exists broadly across these public domains:

1. `vector`
2. `matrix`
3. `lu`
4. `cholesky`
5. `qr`
6. `svd`
7. `eigen`
8. `schur`
9. `polar`
10. `matrix_functions`
11. `orthogonalization`
12. `triangular`
13. `sparse`
14. `batched`
15. `tensor`
16. `iterative`
17. `jacobian`
18. `optimization`
19. `pca`
20. `regression`
21. `stats`

Detailed function-level coverage and result-contract notes live in `docs/ARROW_SUPPORT_MATRIX.md`.

Checkpoint 2 under the concept-first contract is now complete. The remaining question is no longer
"does each concept family have a canonical standalone and `rows-of-X` ingress?" but how
downstream consumers such as `ndatafusion` compose over the now-stabilized Arrow contracts.

## Concept-First Ingress Matrix

This matrix is the release-checkpoint view of `nabled::arrow`:

| Concept family | Standalone ingress | Canonical `rows-of-X` ingress | Current state |
|---|---|---|---|
| Dense vector | one vector object | `FixedSizeList<T>(D)` | Full |
| Ragged vector / multivector | curated through variable-shape tensor workflows | `arrow.variable_shape_tensor` | Full |
| Sparse vector | one sparse vector object | CSR rows | Full |
| Dense matrix | one matrix object | fixed-shape tensor rank-2 carrier | Full |
| Sparse matrix | one sparse matrix object | `ndarrow.csr_matrix_batch` | Full |
| Fixed-shape tensor | one tensor object | `arrow.fixed_shape_tensor` | Full |
| Variable-shape tensor | one ragged tensor object | `arrow.variable_shape_tensor` | Full |
| Complex vector | one complex vector object | first-class complex vector batch carrier | Full |
| Complex matrix | one complex matrix object | first-class complex matrix batch carrier | Full |
| Complex tensor | one complex tensor object | first-class complex tensor batch carrier | Full |

## Current Data Mapping Rules

1. Dense vectors: `PrimitiveArray<T>` -> `ArrayView1<T>`
2. Dense matrices: `FixedSizeListArray` -> `ArrayView2<T>`
3. Sparse CSR:
   - Arrow columns (`List<UInt32>` + `List<T>`) -> `ndarrow::CsrView<T>` -> `nabled::sparse::CsrMatrixView`
   - `ndarrow.csr_matrix` extension -> same route
   - `ndarrow.csr_matrix_batch` extension -> per-row `ndarrow::CsrView<T>` -> per-row `nabled::sparse::CsrMatrixView`
4. Fixed-shape tensors:
   - canonical `arrow.fixed_shape_tensor` storage -> `ArrayViewD<T>`
5. Variable-shape tensors:
   - canonical `arrow.variable_shape_tensor` storage -> per-row `ArrayViewD<T>`
6. Complex dense matrices:
   - nested `FixedSizeList<ndarrow.complex64>(D)` -> `ArrayView2<Complex64>`
7. Complex fixed-shape tensors:
   - `arrow.fixed_shape_tensor<ndarrow.complex64>` -> `ArrayViewD<Complex64>`
8. Complex variable-shape tensors:
   - `arrow.variable_shape_tensor<ndarrow.complex64>` -> per-row `ArrayViewD<Complex64>`

## Scope Boundaries

This integration does not currently provide:

1. Arrow-native decomposition result structs for multi-output workflows; many results remain
   ndarray-native when that is the natural contract.
2. Arrow integration inside lower crates.
3. Automatic admission of new Arrow boundary shapes or result contracts beyond the currently locked
   ones.

## Expansion Rules

Future Arrow surface should expand only when:

1. The underlying `nabled` API is already ndarray-view-native or can be widened generically without
   introducing Arrow-specific code in lower crates.
2. The Arrow result contract is explicit and unsurprising.
3. Docs, tests, and feature-matrix checks are updated in the same change set.
4. The new wrapper moves the Arrow support matrix toward capability parity rather than adding
   low-value overload mirroring.

## Tracking

1. `docs/ARROW_SUPPORT_MATRIX.md` is the direct-ingress coverage ledger for the current public API.
2. Checkpoint 2 is complete under the concept-first standalone / `rows-of-X` matrix above.
3. Future Arrow work should preserve this matrix explicitly instead of expanding wrappers ad hoc.
