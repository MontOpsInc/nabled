# Narrow Interop Workload

Last updated: 2026-03-04

## Purpose

This document introduces a new task workload driven by `narrow`, the zero-copy Arrow-ndarray
bridge being developed alongside nabled. narrow is a sibling crate that enables Arrow-native
systems (DataFusion, Flight, IPC) to leverage nabled's ndarray-native algorithms without
allocation overhead on the bridge path.

narrow does NOT depend on nabled. Both crates share ndarray as their common substrate. However,
several additive changes to nabled would significantly improve the end-to-end zero-copy story
when narrow and nabled are used together.

**Constraint**: None of these changes alter nabled's core identity. nabled remains ndarray-first
with no Arrow dependency. These changes make nabled's own API more general (wider type support,
view-based sparse operations), which benefits all consumers — not just narrow.

**Execution mode for this document**: deterministic and non-optional. The items below are not
exploratory recommendations; they are implementation requirements for this workload.

## Relationship to Existing Tracker

Items here use the `NI-xxx` prefix (Narrow Interop). When prioritized for execution, they
should be added to `docs/EXECUTION_TRACKER.md` as `N-xxx` items with a cross-reference to
the originating `NI-xxx` ID.

All items obey the same quality gates, documentation discipline, and non-negotiable constraints
defined in `AGENTS.md`. No Arrow types enter nabled's public API surface.

## Design Principles (from narrow's architecture)

- **Algebraic, compositional, homomorphic, denotationally sound** — applies to both sides
- **Zero-copy where structurally possible** — narrow bridges Arrow buffers to ndarray views;
  nabled should accept those views without forcing index-type or dtype conversions
- **Explicit allocation boundaries** — if a type conversion must occur, it should be the
  caller's choice, not a hidden cost inside nabled

## Implementation Status

| ID     | Status       | Notes |
|--------|--------------|-------|
| NI-001 | Completed    | Real API surface is now first-class `f32`/`f64` across workspace crates with no cast/wrapper fallback entrypoints. Remaining NI target domains (`eigen`, `matrix_functions`, `sparse`) are fully generic and now have explicit `f32` parity coverage. |
| NI-002 | Completed    | Sparse view model is now first-class with `CsrMatrixView<'a, R, T, C>` and Arrow-native mixed index support (`i32` row offsets + `u32` column indices). |
| NI-003 | Completed    | Sparse owned APIs now have view-first counterparts and owned entrypoints delegate through view-native paths; hidden view->owned sparse materialization helper paths were removed. |

## Items

### NI-001: First-Class f32 Support

**Priority**: High
**Scope**: All crates (`nabled-core`, `nabled-linalg`, `nabled-ml`, facade)
**Breaking**: Yes (API signatures change or widen)

**Problem**: nabled's real-valued API surface is `f64`-only. All public functions take
`&Array2<f64>`, `&ArrayView2<f64>`, etc. Embedding models (and many Arrow producers) commonly
emit `f32` data. Without f32 support, consumers must allocate a widened copy before calling
nabled and a narrowed copy after — two avoidable allocations per operation.

**Goal**: `f32` and `f64` are both first-class across all public APIs.

**Mandated approach**:

1. Use one generic real-float path (Option 1) across the public API surface.
2. No `_f32` duplicate API families.
3. No wrapper/cast entrypoints that hide allocation or type-conversion copies.
4. No retained legacy `f64`-only public signatures once migration is complete.
5. Hand-rolled algorithms must support both `f32` and `f64` directly (generic kernels or
   intentionally duplicated typed internals where mathematically required), with no fallback
   conversion layers.

**Completion contract**:

1. Public real-valued APIs are generic over nabled's real-float trait and compile for both
   `f32` and `f64`.
2. Existing `f64` behavior is preserved (numerical tolerance-aware parity).
3. New `f32` parity tests exist per migrated function family.
4. `just checks` passes and coverage remains >= 90%.
5. Repository shape reflects native dual support, not retrofit wrappers.

### NI-002: CsrMatrixView with Arrow-Native Index Types

**Priority**: High
**Scope**: `nabled-linalg` (sparse module)
**Breaking**: No (additive)

**Problem**: nabled's `CsrMatrix` uses `Vec<usize>` for `row_ptrs` and `col_indices`. Arrow
uses `i32` for List offsets and `u32` for index values. On 64-bit systems, `usize` is 8 bytes
while `i32`/`u32` are 4 bytes — the memory layouts are incompatible for zero-copy.

narrow defines a `CsrView<'a, T>` that borrows `&[i32]` offsets and `&[u32]` indices directly
from Arrow buffers. For this view to flow into nabled's sparse operations without index-type
conversion, nabled needs to accept these types.

**Goal**: Sparse operations accept borrowed CSR data with `i32` offsets and `u32` indices,
in addition to the existing `usize`-based `CsrMatrix`.

**Mandated approach**:

1. Define `CsrMatrixView<'a, R, T, C>` generic over row-pointer and column-index types:
   ```rust
   pub struct CsrMatrixView<'a, R, T, C> {
       pub nrows: usize,
       pub ncols: usize,
       pub row_ptrs: &'a [R],
       pub col_indices: &'a [C],
       pub values: &'a [T],
   }
   ```
   Where `R: CsrIndex` and `C: CsrIndex` (traits bounding `i32`, `u32`, `usize` with checked
   conversion to `usize`).

2. Add `_view` variants of sparse operations that accept `CsrMatrixView` (consistent with
   nabled's existing `_view` pattern for dense operations).

3. Internally, sparse kernels that need `usize` indexing can convert on the fly within tight
   loops (the conversion is a register widening, not an allocation).

**Completion contract**:

1. Existing `CsrMatrix` APIs remain valid.
2. `CsrMatrixView` is first-class for sparse APIs.
3. Tests verify `CsrMatrixView` support with mixed `i32`/`u32` index types.
4. `just checks` passes and coverage remains >= 90%.

### NI-003: View-Accepting Sparse Operations

**Priority**: Medium
**Scope**: `nabled-linalg` (sparse module)
**Breaking**: No (additive)
**Depends on**: NI-002

**Problem**: Sparse operations currently take owned `CsrMatrix`. For zero-copy from Arrow,
operations should also accept borrowed `CsrMatrixView`.

**Goal**: Every sparse operation that takes `&CsrMatrix` also has a `_view` variant that
accepts `&CsrMatrixView` with equivalent semantics.

**Mandated approach**: Follow the existing `_view` pattern used throughout nabled's dense API
(`matvec_view`, `decompose_view`, etc.). Each `_view` variant accepts the view type from
NI-002 and produces owned results (consistent with nabled's output-is-always-owned convention).

**No partial rollout**:

1. This applies everywhere in sparse where `&CsrMatrix` is accepted today.
2. No allowlist of selected functions; coverage is exhaustive for the sparse module surface.
3. No compatibility re-exports or wrapper shims.

**Completion contract**:

1. Every sparse API that accepts `&CsrMatrix` has a corresponding `_view` variant.
2. `_view` variants have parity/error-path tests against owned equivalents.
3. `just checks` passes and coverage remains >= 90%.

### NI-004: Removed From This Workload

NI-004 (complex Arrow representation assessment) is out of scope for nabled and belongs to
narrow-side design work. It is intentionally not tracked as an execution item in this document.

---

## Execution Priority

| ID     | Description                            | Priority | Blocking narrow? |
|--------|----------------------------------------|----------|------------------|
| NI-001 | First-class f32 support                | High     | No               |
| NI-002 | CsrMatrixView with Arrow-native types  | High     | No               |
| NI-003 | View-accepting sparse operations       | Medium   | No               |

**None of these items block narrow's development.** narrow uses ndarray directly and defines
its own types. These items improve the zero-copy end-to-end story but are not prerequisites.

## Integration with Existing Workload

These items can be interleaved with existing `N-*` priorities.

- **NI-001** (f32): Completed.
- **NI-002** (CsrMatrixView): Completed.
- **NI-003** (sparse views): Completed.

## Reference

- narrow's interop tracker: `narrow/docs/NABLED_CHANGES.md`
- narrow's architecture: `narrow/docs/architecture.md`
- narrow's performance contracts: `narrow/docs/PERFORMANCE_CONTRACTS.md`
